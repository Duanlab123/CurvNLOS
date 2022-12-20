close all
clear all

addpath('./util');
load('bowling.mat');                % load groundtruth
figure(1);draw3D(scene,0.5,0.1,1);title('goundtruth'); drawnow % show groundtruth
scene = permute(scene,[3 2 1]);

N = size(scene,1);                  % pixel number of x,y
bin = size(scene,3);                % pixel number of time or z axis
samp = 4;                           % scan points samp x samp
mask = definemask(N,samp); 
filename = strcat('scan',num2str(samp));

%% simulation setting
bin_resolution = 32e-12;            % Time resolution
wall_size = 1;                      
width = wall_size/2;                % scan range -width to width (unit:m)
c = 3*10^8;        % speed of light
range = bin.*c.*bin_resolution;
slope = width./range;
[mtx,mtxi] = resamplingOperator(bin);
mtx = full(mtx);
mtxi = full(mtxi);
psf = definePsf(N,width,bin,bin_resolution,c);          % refer to lct reconstruction

FWHM = 60e-12;                      % FWHM of system temporal jitter
sigmaBin = FWHM/bin_resolution/2/sqrt(2*log(2));
jit = normpdf(-3:3, 0, sigmaBin);
jit = reshape(jit, [1 1 7]);        % temporal blur

A = @(x) forwardmodel(x,psf,mtxi,jit,N,bin,mask);       % forwardmodel of NLOS imaging
AT = @(x) x.*mask;                                      % not used
%%  generate simulated data
y = A(scene);

y = permute(y,[3 2 1]);
grid_z = repmat(linspace(0,1,bin)',[1 N N]);
y = y./(grid_z.^4+0.0001);
y = permute(y,[3 2 1]);  % z^4 dacay

y = random('Poisson',y*1000)/1000;     % poisson noise
y = y.*mask;
% [phasor,lct,fk] = lctdatacompare(y,N,bin,mask,width,bin_resolution,samp);
% figure(1);draw3D(phasor,0.5,0.1,1);title('phasor');
% figure(2);draw3D(lct,0.5,0.1,1);title('lct');
% figure(3);draw3D(fk,0.5,0.1,1);title('fk');
% [phasoraccuracy,phasorrmse] = depthevaluate(phasor)
% [lctaccuracy,lctrmse] = depthevaluate(lct)
% [fkaccuracy,fkrmse] = depthevaluate(fk)
% [psp,ssp] = psnr1(phasor) %single
% [psl,ssl] = psnr1(lct) %double
% [psf,ssf] = psnr1(fk)  %double

box = drawbox(0.6,3);
y = permute(y,[3 2 1]);
grid_z = repmat(linspace(0,1,bin)',[1 N N]);
y = y.*(grid_z.^4);
lambdad = 300; lambdax = 1; mu1 = 1; mu2 = 8e2; mu3 = 2;ad = 1e-4; bd = 1.2e-2; au = 6e-4;bu = 3e-4;mu = 0.1;au0 = 1e-5;bu0 = 1e-5;
tolerance = 1e-6;

tic
t1=clock;
% [x,iter,objectiveout] = dualdomain(y,psf,jit,samp,lambdad,lambdax,mu,mu1,mu2,mu3,au0,bu0,au,bu,ad,bd,tau,tolerance);
[x,iter,objectiveout] = imagedomain(y,psf,jit,box,samp,lambdad,lambdax,mu,mu2,mu3,au0,bu0,ad,bd,tau,tolerance);
result = reshape(mtxi*x(:,:),[bin N N]);
x = result.*(grid_z.^0.5);    % convert signal to t domain
t2=clock;
figure(2);draw3D(x,0.5,0.1,1);title('spiral 3D0');
[xaccuracy0,xrmse0] = depthevaluate(x)
[ps0,ss0] = psnr1(x)
xb = smooth3(x,'b');             % smooth
figure(3);draw3D(xb,0.5,0.1,1);title('spiral 3Db');
[xaccuracy,xrmse] = depthevaluate(xb)
[ps,ss] = psnr1(xb)
t=etime(t2,t1);
%% Used Function
function mask = definemask(N,samp)
    xx = linspace(1,N,samp);
    yy = linspace(1,N,samp);
    mask = zeros(N,N); 
    for i = 1:samp
        for j = 1:samp
            mask(round(xx),round(yy)) = 1;
        end
    end
end


function psf = definePsf(N,width,bin,timeRes,c)     % refer to lct reconstruction
    linexy = linspace(-2*width,2*width,2*N-1);            % linspace of scan range
%     linexy = linspace(-2*width,2*width,2*N);    
    range = (bin*timeRes*c/2)^2;                    % total range of t^2 domain: bin^2*timeRes^2
    gridrange = bin*(timeRes*c/2)^2;                % one grid in t^2 domain: bin*timeRes^2
    [xx,yy,squarez] = meshgrid(linexy,linexy,0:gridrange:range);
    blur = abs((xx).^2+(yy).^2-squarez+0.0000001);
    blur = double(blur == repmat(min(blur,[],3),[ 1 1 bin+1 ]));
    blur = blur(:,:,1:bin);                               % generate light-cone
    psf = zeros(2*N-1,2*N-1,2*bin); 
%     psf = zeros(2*N,2*N,2*bin); 
%     psf(2:2*N,2:2*N,bin+1:2*bin) = blur;
    psf(:,:,bin+1:2*bin) = blur;                          % place it at center
end

function Ax = forwardmodel(x,psf,mtxi,jit,N,bin,mask)   
    d = convn(x,psf,'same');             % signal response in t^2 domain
    pd = permute(d,[3 2 1]);             % convert signal to t domain
    mpd = reshape(mtxi*pd(:,:),[bin N N]);
    b = permute(mpd,[3 2 1]);
    Ax = convn(b,jit,'same');            % convolution with jitter
    Ax = Ax.*mask;
end
  

function [mtx,mtxi] = resamplingOperator(M)   % refer to lct reconstruction
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


function [accuracy,rmse] = depthevaluate(vol)  
    load('bowlingscene.mat');
    for i = 1:64
        for j = 1:64
            [sceneref(i,j),scenedep(i,j)] =  max(squeeze(scene(:,i,j)));
        end
    end
    sceneref = sceneref/max(sceneref(:));
    indscene = sceneref>0.1;
    numind = sum(indscene(:));
  
    for i = 1:64
        for j = 1:64
            [front(i,j),dep(i,j)] =  max(squeeze(vol(:,i,j)));
        end
    end
    front = front/max(front(:));
    indfront = front>0.1;
    accuracy = 1-sum(sum((indfront-indscene).^2,1),2)/4096;
    rmse = sqrt(norm(dep(indscene)-scenedep(indscene))/numind)*0.48;
end
function box = drawbox(boxsize,dim)
    switch dim
        case 3
            [xx,yy,zz] = meshgrid(-5:5,-5:5,-5:5);
            box = exp(-(xx.^2+yy.^2+zz.^2)/(boxsize^2));
            box = box/sum(box(:));
        case 2
            [xx,yy] = meshgrid(-5:5,-5:5);
            box = exp(-(xx.^2+yy.^2)/(boxsize^2));
            box = box/sum(box(:));
            box = reshape(box,[1,11,11]);
            
    end
end
function [ps,ss] = psnr1(vol)  

    x = vol;
    x = x/max(x(:));
    x1 = squeeze(max(abs(x),[],1));
    load('bowlingscene.mat');
    xref = single(scene);%double(scene);%single(scene);
%     xref = scene;
    xref = xref/max(xref(:));
    xref1 = squeeze(max(abs(xref),[],1));
    ps = psnr(x1,xref1);
    ss = ssim(x1,xref1);
end

    
