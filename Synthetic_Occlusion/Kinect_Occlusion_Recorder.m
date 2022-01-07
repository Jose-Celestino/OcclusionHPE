%% ACQUIRE KINECT DATA

% INITIALIZE KINECT RGB AND DEPTH SENSORS.
colorDevice = imaq.VideoDevice('kinect',1);
depthDevice = imaq.VideoDevice('kinect',2);

fprintf('Start \n');
pause(5);

% Record multiple color, depth frames and point clouds.
for i = 1:20    
   colorImage = step(colorDevice);  % Capture RGB image.
   depthImage = step(depthDevice);  % Capture Depth image.
 
   % Returns a point cloud using depth and color data.
   ptCloud = pcfromkinect(depthDevice,depthImage,colorImage); 
   
   % Store all data for respective frame.
   points{i} = ptCloud;
   RGB{i} = colorImage;
   Depth{i} = depthImage;
   
   % If point cloud visualization is desired, uncomment.
   %player = pcplayer(ptCloud.XLimits,ptCloud.YLimits,ptCloud.ZLimits,...
	%'VerticalAxis','y','VerticalAxisDir','down');
   %view(player,ptCloud);
   
   pause(5); %pause in the recording, if necessary to move occlusion object.
   disp(i);
end

fprintf('End');
release(colorDevice); % Free RGB sensor.
release(depthDevice); % Free Depth sensor.


%% MANUAL INDIVIDUAL KINECT DATA RECORDING
% Manually record RGB, Depth and Point Cloud data for individual frames.

% INITIALIZE KINECT RGB AND DEPTH SENSORS.
colorDevice = imaq.VideoDevice('kinect',1);
depthDevice = imaq.VideoDevice('kinect',2);

i = 1;

pause(2);
   
colorImage = step(colorDevice);  % Capture RGB image.
depthImage = step(depthDevice);  % Capture Depth image.

% Returns a point cloud using depth and color data.
ptCloud = pcfromkinect(depthDevice,depthImage,colorImage);
   
% Store all data for respective frame.
points{i} = ptCloud;
RGB{i} = colorImage;
Depth{i} = depthImage;
 
% If point cloud visualization is desired.
player = pcplayer(ptCloud.XLimits,ptCloud.YLimits,ptCloud.ZLimits,...
    'VerticalAxis','y','VerticalAxisDir','down');
 
view(player,ptCloud);

% Free sensors.
release(colorDevice);
release(depthDevice);

%% SAVE THE DATA IN .MAT FILE

Filename = 'Occlusion.mat'; 
save(Filename, 'points', 'RGB', 'Depth');


