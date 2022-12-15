function data = dataproduce(scene,fpsf,mtx,mtxi)

N = size(scene,1);        % Spatial resolution of data
M = size(scene,3);        % Temporal resolution of data
scene = permute(scene,[3 2 1]);
grid_z = repmat(linspace(0,1,M)',[1 N N]);
grid_z(1,:,:) = 1;
tscene = zeros(2.*M,2.*N,2.*N);
tscene(1:end./2,1:end./2,1:end./2)  = reshape(mtx*scene(:,:),[M N N]);
tdata = ifftn(fftn(tscene).*fpsf);
tdata = tdata(1:end./2,1:end./2,1:end./2);

% Step 4: Resample depth axis and clamp results
data  = reshape(mtxi*tdata(:,:),[M N N]);
data = data./(grid_z.^4);
data = permute(data,[3 2 1]);
data = max(data,0);