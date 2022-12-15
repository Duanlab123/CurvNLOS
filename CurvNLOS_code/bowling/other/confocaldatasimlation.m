function sig_in = confocaldatasimlation(scene,width,N,M,timeRes,T)

% (x,y,z) for the laser/detection points on the visible wall
lx = linspace(-width, width, N);
ly = linspace(-width, width, N);                      % set up (x,y) coordinates for a 1m*1m scanning area on the visible wall
                                                   % The number of the obtained pixels is 64*64 = 4096

laserPoints = zeros(N*N, 3);
detectPoints = zeros(N*N, 3);    

point_index = 1;
for i = 1 : N
    for j = N : -1 : 1
        laserPoints(point_index, :) = [lx(j), ly(i), 0];
        detectPoints(point_index,:) = [lx(j), ly(i), 0];
        point_index = point_index + 1;
    end
end
clear point_index;                                 % give the coordinates point by point
                                                   % Here, as for the confocal model,these two matrice are the same
                                                   % Also, for simplicity, suppose z=0 for the points on the visible wall 

point_index = 1;
for ii = 1 : N
    for jj = 1 : N
        if(scene(ii,jj) ~= 0)
            objPoints(point_index,:) = [lx(jj), ly(ii), scene(ii,jj)];
            point_index = point_index + 1;
        end
    end
end                                                % Here,for simplicity, Z_true only has two different depth values, 0 m and 0.7 m, 
                                                   % also, we suppose the reflectivity matrix alpha has a value of 1 where the depth is 0.7 m and of 0 where the depth is 0 m
                                                   % Besides, the objPoints matrix only stores the positions for the pixels with the reflectivity of 1
                                                   

c = 3e8;                                           % unit:m/s, the speed of light
lamda = 1550e-9;                                   % unit:m, the wavelength of the laser
P = 1e-5;                                          % unit:W, the power of the laser
h = 6.626e-34;                                     % unit:J*s,  planck constant
emitted_photons_second = P*lamda/h/c;              % unit:counts/s, emitted photon counts per second



FWHM = 150*10^(-12);
pulseWidth = FWHM/2/sqrt(2*log(2));                % FWHM of the laser is set to be 60ps

A_obj = 1e-4;                                      % unit:m^2, the area of one single pixel of the object 
A_FOV = 1e-2;                                      % unit:m^2, The area of the detect pixel of the visible wall
A_tel = 7e-2;                                      % unit:m^2, the telescope aperture
eff = 0.01;                                        % system efficiency
ref = 0.1;                                         % reflectivity of the visible wall

loss0 = A_FOV*A_obj*A_tel*eff*ref^2;               % In my opinion, the above five parameters can be aggregated, 
                                                   % and for each scanning points and pixel on the target, they are the same. 
                                                   % They are not really considered and are just used to control the intensity of the signal
                                                  
% begin the simulation

signal_detection = zeros(size(laserPoints,1), M);                               % initialize the signal histogram with the size of 4096*2048
                                                                                   % The number of the pixels is 4096 and of the time bins is 2048 

for i = 1 : length(laserPoints)                    
    for j = 1 : length(objPoints)                                                  % for each laser point, calculating all the target pixels
        
        r2(j) = norm(laserPoints(i,:) - objPoints(j,:));
        r3(j) = norm(detectPoints(i,:) - objPoints(j,:));                        % calculate the range between the laser/detection points and target pixels
        theta2(j) =  objPoints(j,3)/r2(j);   
        theta3(j) =  objPoints(j,3)/r3(j); 
    end
    
    loss = loss0./(2*3.14*r2.^2)./(2*3.14*r3.^2).*theta2.^2.*theta3.^2;                               
    returned_number = loss .* emitted_photons_second * T;                                         
    t2 = (r2 + r3)./c;                                                          % the time of flight information 
    returned_number = random('poisson',returned_number); 
    for j = 1 : length(objPoints) 
        returned_photons = round(normrnd(t2(j),pulseWidth,returned_number(j),1)/timeRes); % take the pulse width into consideration
        for k = 1 : length(returned_photons)
            if( 1 <= returned_photons(k) && returned_photons(k) <= M )
                signal_detection(i, returned_photons(k)) = signal_detection(i, returned_photons(k)) + 1;
            end
        end
    end
end

%signal_detection = random('poisson', signal_detection);      
sig_in = reshape(signal_detection, [ N N M ]);  % transform the size of the detection matrix (4096*2048) to 64*64*2048

% So far, we have finished the whole simulation. 
%plot(sum(signal_detection,1));   % This shows the third bounce signal (sum the 4096 pixels)
end





