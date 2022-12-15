function [x,ii,objective] = imagecompress(d0,B,psf,jit,box,samp,tau,tolerance) 
    % display plots
    plots = 0;
    %d0:N N bin  
    %d0 : bin N N  psf : 2bin 2N 2N
    d0 = gpuArray(single(d0));
    jit = gpuArray(single(jit));
    box = gpuArray(single(box));
    [bin, N, N] = size(d0);
    max_iters = 200;
    mask = definemask(N,samp);
    mask = reshape(mask,[1,N,N]);
    mask = repmat(mask,[bin 1 1]);
    bin_resolution = 32e-12;            % Time resolution
    wall_size = 1; %1 
    snr = 0.1;
    width = wall_size/2;                % scan range -width to width (unit:m)
    c = 3*10^8;        % speed of light
    range = bin.*c.*bin_resolution;  %
    slope = width./range;
    z_start = 0; 
    z_stop = 1./slope;
    S = 1;
    converged = 0;
    z_offset = 30;
    [mtx,mtxi] = resamplingOperator(bin);
    mtx = full(mtx);
    mtxi = full(mtxi);
    mtx = gpuArray(single(mtx));
    mtxi = gpuArray(single(mtxi));
    psf = gpuArray(single(psf)); 
    
    % utility functions
    trim_array = @(x) x(S*bin/2+1:end-S*bin/2, N/2+1:end-N/2, N/2+1:end-N/2);
    pad_array = @(x) padarray(x, [S*bin/2, N/2, N/2]);
    square2cube = @(x) reshape(x, [],N,N);
    cube2square = @(x) x(:,:);
    vec = @(x) x(:);
    Ffun = @(x)  fftn(x);
    Ftfun = @(x) ifftn(x);
    
    pad_array1 = @(x) padarray(x, [bin/2, N/2-1, N/2-1],'pre');
    pad_array2 = @(x) padarray(x, [bin/2, N/2+1, N/2+1],'post');
    trim_array1 = @(x) x(bin/2+1:3*bin/2, N/2:3*N/2-1, N/2:3*N/2-1);

    % gradient kernels
    d2 = [0 0 0; 0 1 -1; 0 0 0];
    d2 = padarray(d2, [0,0,1]);
    d1 = [0 0 0; 0 1 0; 0 -1 0];
    d1 = padarray(d1, [0,0,1]);
    d3 = zeros(3,3,3);
    d3(2,2,2) = 1; d3(2,2,3) = -1;

    % operator functions
    p2o = @(x) psf2otf(x, [N,N,bin]);
    d1FT = p2o(d1);
    d2FT = p2o(d2);
    d3FT = p2o(d3);
    
    pad_arrayx = @(x) padarray(x, [2*bin-1,2*N-2,2*N-2],'post');
    pad_arraypsf = @(x) padarray(x, [bin-1,N-1,N-1],'post');
    trim_arrayx = @(x) x(bin:2*bin-1, N:2*N-1, N:2*N-1);
    trim_arrayjit = @(x) x(4:3+bin,:,:);
    psf = permute(psf,[3 2 1]);
    psf1 = padarray(psf,[0,1,1],'pre');
    psfFT = fftn(psf1);
    psfFT1 = fftn(padarray(flip(flip(flip(psf,1),2),3),[0,1,1],'pre'));
    jit = permute(jit,[3 2 1]);
    jitFT = fftn(padarray(jit,[bin-1,N-1,N-1],'post'));

    A = @(x) real(trim_arrayjit(Ftfun(jitFT.*Ffun(padarray(square2cube(mtxi*cube2square(trim_array1(real(ifftshift(Ftfun(psfFT .* Ffun(pad_array2(pad_array1(x))))))))),[6 0 0],'post'))))).* mask;
    AT = @(x) trim_array1(real(ifftshift(Ftfun(psfFT1 .* Ffun(pad_array2(pad_array1(square2cube(mtx*cube2square(x)))))))));

    shrinkage = @(a,kappa) repmat(max(0, 1-kappa./sqrt(a(:,:,:,1).^2 + a(:,:,:,2).^2 + a(:,:,:,3).^2)),1,1,1,3).*a;

    
    % ---- For acceptance criterion ---
    alphainit = 1;
    alphamin = 1e-30;
    alphamax = 1e15;
    acceptalphamax = alphamax;
    acceptmult = 2;%8,2
    acceptdecrease = 0.1;
    acceptpast = 10;
    ii = 1;
%     tolerance = 1e-5;

    xinit = d0*0;
    x = xinit;    
    
    xprevious = x;
    Ax = A(x);
    alpha = alphainit;
    Axprevious = Ax;
    grad = AT(Ax-d0);
    objective = [];
    objective = zeros(max_iters+1,1);
    objective = gpuArray(single(objective));
    objective(1) = computeobjective(x,d0,Ax,tau,'gaussian',1e-10,'canonical',[]);

    while (ii <= max_iters) && not(converged)  
        
        tic
        past = (max(ii-1-acceptpast,0):ii-1) + 1;
        maxpastobjective = max(objective(past));
        accept = 0;
        while (accept == 0)
            % --- Compute the step, and perform Gaussian 
            %     denoising subproblem ----
            dx = xprevious;
            z = xprevious - grad./alpha;
            x = max(z - tau./alpha, 0.0);
%             if mod(ii,10) == 0
%                 x(:,:,:) = convn(x(:,:,:),box,'same'); 
%             end
            realgate = max(x(:))*0.05;
            x = x.*(abs(x)>realgate); % set small value to zero
            dx = x - dx;
            Adx = Axprevious;
            Ax = A(x);
            Adx = Ax - Adx;
            normsqdx = sum( dx(:).^2 );
            
            objective(ii + 1) = computeobjective(x,d0,Ax,tau,'gaussian',1e-10,'canonical',[]);
                    
            if ( objective(ii+1) <= (maxpastobjective ...
                            - acceptdecrease*alpha/2*normsqdx) ) ...
                            || (alpha >= acceptalphamax);
                accept = 1;
            end
        acceptalpha = alpha;  % Keep value for displaying
        alpha = acceptmult*alpha;
        end
        grad = AT(Ax-d0);
        gamma = sum(Adx(:).^2);
        if gamma == 0
            alpha = alphamin;
        else
            alpha = gamma./normsqdx;
            alpha = min(alphamax, max(alpha, alphamin));
        end
        xprevious = x;
        Axprevious = Ax; 
        
        converged = (abs(objective(ii + 1)-objective(ii))./abs(objective(ii)) <= tolerance);
%         converged = ((sum(dx(:).^2)./sum(x(:).^2)) <= tolerance^2);
        ii = ii + 1;
        
        times(ii) = toc;   
        
        fprintf('Iteration: %d, Elapsed: %.01f\n', ii, times(ii));

        vol(:,:,:) = gather(abs(x(:,:,:)));
        figure(1);draw3D(gather(x),0.5,0.1,1);title('x'); drawnow;
       
    end
    x = gather(vol);
end

function objective = computeobjective(x,y,Ax,tau,noisetype,logepsilon,...
    penalty,varargin)
% Perhaps change to varargin 
% 1) Compute log-likelihood:
switch lower(noisetype)
    case 'poisson'
        precompute = y.*log(Ax + logepsilon);
        objective = sum(Ax(:)) - sum(precompute(:));
    case 'gaussian'
        objective = sum( (y(:) - Ax(:)).^2)./2;
end
% 2) Compute Penalty:
switch lower(penalty)
    case 'canonical'
        objective = objective + sum(abs(tau(:).*x(:)));
    case 'tv'
        objective = objective + tau.*tlv(x,'l1');
end
end

