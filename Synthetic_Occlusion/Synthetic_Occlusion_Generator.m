%% SYNTHETIC OCCLUSION GENERATOR

%Code is split in sections so that the user can run the sections he needs.

%CODE STRUCTURE
% 1. Load Occlusions (.mat file);
% 2. Load Original Dataset to be occluded;
% 2.1. Random Fixed Occlusions Indices (optional)
% 3. Determining Threshold Distance for First Frame
% 3.1. generate output cluster data (optional)
% 3.2. Plot result cluster (optional)
% 4. Iterate and Generate New Synthetically Occluded Dataset
%% 1. Load Occlusion .mat File 

%load('occlusion.mat');
Fixed_Occlusion_Flag = 0; %flag to indicate we the user desires fixed occ.

%% 2. Load Images in Dataset 

images = dir('Occluded_faces\*.jpg');
images = natsortfiles(images);

%% 2.1. Random Fixed Occlusions Indices (optional)
% If you desire to apply the same random occlusions to more than one
% dataset run this section.

Fixed_Occlusion_Flag = 1; %flag to indicate we the user desires fixed occ.
matrix_index = zeros(length(images),1);

% A fixed random occlusion for each dataset image
for i = 1:length(images)
    
    %first frame from .mat file must be non-occluded, we desire index from
    %2 to the number of occlusions.
    ind = randperm(length(points)-1,1)+1; 
    matrix_index(i,:) = ind;
    
end


%% 3. Determining Threshold Distance for First Frame

% Initialize structure to store occluded images if necessary.
%occluded_images = cell(1, length(images));

% Face Detection (no occlusion)
rgb1 = points{1}.Color; %rgb data for frame 1
xyz1 = points{1}.Location(); %depth data for frame 1
%double(reshape(points{1}.Location(), [480*640 3]))'; 

% Apply MTCNN face detector to image                                                       
[rect, scores, landmarks] = mtcnn.detectFaces(rgb1);

% Limits of Bounding Box(rect)
ymin = rect(2);
ymax = rect(2)+ rect(4);
xmin = rect(1);
xmax = rect(1) + rect(3);

% If you desire to broaden or reduce the border of the face detector,
% uncomment the lines below
%k = 0.2;
% x_min = xmin - k * abs(xmax - xmin);
% y_min = ymin - k * abs(ymax - ymin);
% x_max = xmax + k * abs(xmax - xmin);
% y_max = ymax + k * abs(ymax - ymin);

%if you desire to use the bounding box of the face detector
x_min = xmin;
y_min = ymin;
x_max = xmax;
y_max = ymax;


rect_new = [x_min, y_min, x_max-x_min, y_max-y_min]; 

orig_face_xyz = xyz1(ceil(y_min):ceil(y_max),ceil(x_min):ceil(x_max),1:3);
orig_face_xyz = reshape(orig_face_xyz, [size(orig_face_xyz,1)*size(orig_face_xyz,2) 3]);

% CLUSTERING TO AVOID OUTLIERS IN DISTANCE SEPARATION
addpath('utils');
[class,~] = dbscan(orig_face_xyz,2,[]);

%biggest cluster will correspond to the face 
for j=1:max(class)
    tam = size(find(class == j));
    counter(j) = tam(:,2);
end

% find points of biggest cluster (face cluster)
big_class = find(counter == max(counter));
cara_index = find (class == big_class);
face = orig_face_xyz(cara_index,:); % face xyz points

% Threshold distance that identifies occlusions
distance_sep = min(face(:,3)); 

%% 3.1. generate output cluster data (optional)
% cell_cluster={};x=orig_face_xyz(:,1);y=orig_face_xyz(:,2);z=orig_face_xyz(:,3);
% for i=1:max(class)
%     cluster_i=[x(class==i),y(class==i),z(class==i)];%call x,y,z coord of all points which is belong the same cluster
%     cell_cluster{end+1} = cluster_i;%this is (1xk)cell. where k=number of cluster
% end
%% 3.2. Plot result cluster (optional)
% Legends = {};figure
% for i=1:max(class)
%     hold on,view(3);Legends{end+1} = ['Cluster #' num2str(i)];
%     scatter3(x(class==i),y(class==i),z(class==i),1,'.');
% end
% legend(Legends);legend('Location', 'NorthEastOutside');%show legend for figure
% title(['DBSCAN Clustering']);%show title
% hold on;scatter3(x(class==-1),y(class==-1),z(class==-1),21,'*');%plot outlier(noise data)

%% 4. Iterate and Generate New Synthetically Occluded Dataset

% For each dataset image
for z = 1:length(images)
    image_data = images(z);
    image_data = imread(fullfile(image_data.folder,image_data.name));
    
    
    if Fixed_Occlusion_Flag == 1 %If fixed occlusions are desired
        i = matrix_index(z,1);
    else                            
        i = randperm(length(points)-1,1)+1; %select random occlusion
    end
    

    
    %depth image
    xyz = points{i}.Location();
    
    %color image
    rgb = points{i}.Color;
    
    %crop rgb and depth within detection box
    frame_bbox = rgb(ceil(y_min):ceil(y_max),ceil(x_min):ceil(x_max),1:3);
    face_rgb = reshape(frame_bbox, [size(frame_bbox,1)*size(frame_bbox,2) 3]);
    face_xyz = xyz(ceil(y_min):ceil(y_max),ceil(x_min):ceil(x_max),1:3); 
    face_xyz = reshape(face_xyz, [size(face_xyz,1)*size(face_xyz,2) 3]);
    
    
    %select points closer than threshold distance (occlusion points)
    valid = face_xyz(:,3) < distance_sep; 
    face_xyz = face_xyz(valid,:);

    

    
    
    %crop image to face box

    

    
    %we only want the obstacle to overlay on the dataset image
    face_rgb(~valid,:) = 0;
    %reshape vector to face box dimension
    occlusion = reshape(face_rgb, [size(frame_bbox,1), size(frame_bbox,2), 3]);
    
    
    %rescales object image size to dataset image size
    occlusion_resize = imresize(occlusion,[size(image_data,1),size(image_data,2)]); 
   
    imcropped_vector = reshape(image_data,[size(image_data,1)*size(image_data,2),3]); %matrix of dataset image
    occlusion_resize_vector = reshape(occlusion_resize,[size(image_data,1)*size(image_data,2),3]); %matrix of rescaled object image
    
  
    %finds nonzero elements(image occlusion pixels);
    occlusion_index = find(occlusion_resize_vector(:,1)); 
    %superimposes occlusion in original dataset image
    imcropped_vector(occlusion_index,:) = occlusion_resize_vector(occlusion_index,:);
    
    %reshapes synthetic occluded image matrix to original image shape
    data_occluded = reshape(imcropped_vector,[size(image_data,1),size(image_data,2),3]);
      
    if ~(mod(z,1000))
        fprintf('Occlusion %d \n',z);
    end
    
    %writes image in destiny folder
    folder = 'Destiny_Folder\';
    FILENAME = string(strcat(folder, 'oclusion', num2str(z), '.jpg'));
    imwrite(data_occluded,FILENAME);
    
    
end
