function imslice(vol,width,range)

vol = permute(vol,[1 3 2]);
tic_z = linspace(0,range./2,size(vol,1));
tic_y = linspace(-width,width,size(vol,2));
tic_x = linspace(-width,width,size(vol,3));

figure('pos',[10 10 900 300]);

subplot 224;
imagesc(tic_x,tic_y,squeeze(max(vol,[],1)));
title('Front view');
set(gca,'XTick',linspace(min(tic_x),max(tic_x),3));
set(gca,'YTick',linspace(min(tic_y),max(tic_y),3));
xlabel('x (m)');
ylabel('y (m)');
colormap('gray');
axis square;

subplot 222;
imagesc(tic_x,tic_z,squeeze(max(vol,[],2)));
title('Top view');
set(gca,'XTick',linspace(min(tic_x),max(tic_x),3));
set(gca,'YTick',linspace(min(tic_z),max(tic_z),3));
xlabel('x (m)');
ylabel('z (m)');
colormap('gray');
axis square;

subplot 223;
imagesc(tic_z,tic_y,squeeze(max(vol,[],3))')
title('Side view');
set(gca,'XTick',linspace(min(tic_z),max(tic_z),3));
set(gca,'YTick',linspace(min(tic_y),max(tic_y),3));
xlabel('z (m)');
ylabel('y (m)');
colormap('gray');
axis square;


x = volumeview(vol);
x_permute = permute(x(:,end:-1:1,:), [1 3 2]);
subplot 221; vol3d('CData', x_permute); 
axis image; xlabel('z'); ylabel('x'); zlabel('y'); colorbar;
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
set(gca,'ztick',[])
set(gca,'zticklabel',[])

% set(gca,'xtick',[])
% set(gca,'xticklabel',[])
% set(gca,'ytick',[])
% set(gca,'yticklabel',[])
% set(gca,'ztick',[])
% set(gca,'zticklabel',[])
set(gca, 'Xcolor', 'none');
set(gca, 'Ycolor', 'none');
set(gca, 'Zcolor', 'none');

view(20,15) %¸ü»»½Ç¶È
%view(135,35);

set(gca, 'color', 'k');
colormap gray;

end