function [phasor,lct,fk] = lctdatacompare(y,N,M,mask,width,bin_resolution,samp)
    data = y;    
    addpath('other');
    mask = definemask(N,samp);
    xx = round(linspace(1,N,samp));
    yy = round(linspace(1,N,samp));
    
    [sxx,syy] = meshgrid(xx,yy);
    [xxx,yyy] = meshgrid(1:N,1:N);
    for i = 1:M
        tmpdata = data(:,:,i);
        data(:,:,i)=interp2(sxx,syy,reshape(tmpdata(logical(mask)),[samp,samp]),xxx,yyy,'linear');%¶þÎ¬²åÖµ
    end
%     save('y32inpaninte.mat','data');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    c    = 3e8;%3e8;    % Speed of light (meters per second)  
    range = M.*c.*bin_resolution; % Maximum range for histogram
    data = permute(data,[3 2 1]);
    psf = definePsf(N,M,width./range);
    fpsf = fftn(psf);
    [mtx,mtxi] = resamplingOperator(M);
    mtx = full(mtx);
    mtxi = full(mtxi);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    phasor = PHASOR(data,M,N,fpsf,mtx,mtxi,2*width,bin_resolution,2);%3,3.5,5,8
    lct = LCT(data,M,N,fpsf,mtx,mtxi,0.01);%0.4,0.4,0.4,0.1
    %fbp = FBP(data,M,N,fpsf,mtx,mtxi);
    fk =  FK(data,M,N,range,width);
    
%     phasor(end-10:end, :, :) = 0; 
%     fk(end-10:end, :, :) = 0;
%     lct(end-10:end, :, :) = 0;
    
%     phasor(end-120:end, :, :) = 0; 
%     fk(end-120:end, :, :) = 0;
%     lct(end-120:end, :, :) = 0;
        
    %threedshow(lct,range,width);
    %threedshow(fbp,range,width);
    %threedshow(fk,range,width);
    %threedshow(phasor,range,width);
    
end










function psf = definePsf(U,V,slope)
    % Local function to compute NLOS blur kernel
    x = linspace(-1,1,2.*U);
    y = linspace(-1,1,2.*U);
    z = linspace(0,2,2.*V);
    [grid_z,grid_y,grid_x] = ndgrid(z,y,x);

    % Define PSF
    psf = abs(((4.*slope).^2).*(grid_x.^2 + grid_y.^2) - grid_z);
    psf = double(psf == repmat(min(psf,[],1),[2.*V 1 1]));
    psf = psf./sum(psf(:,U,U));
    psf = psf./norm(psf(:));
    psf = circshift(psf,[0 U U]);
end

function [mtx,mtxi] = resamplingOperator(M)
 % Local function that defines resampling operators
     mtx = sparse([],[],[],M.^2,M,M.^2);
     
     x = 1:M.^2;
     mtx(sub2ind(size(mtx),x,ceil(sqrt(x)))) = 1;
     mtx  = spdiags(1./sqrt(x)',0,M.^2,M.^2)*mtx;
     mtxi = mtx';
     
     K = log(M)./log(2);
     for k = 1:round(K)
         mtx  = 0.5.*(mtx(1:2:end,:)  + mtx(2:2:end,:));
         mtxi = 0.5.*(mtxi(:,1:2:end) + mtxi(:,2:2:end));
     end
end
