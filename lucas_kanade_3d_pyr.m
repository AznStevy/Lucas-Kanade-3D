%% Main function for cell-tracking
clear all; close all;
warning('off','all')
addpath(genpath(pwd));

%% Load volumes
vol_1 = load_volume('data_2/SPIMA-0.tif');
vol_2 = load_volume('data_2/SPIMA-1.tif');

%% Try creating a max projection along a single dimension to see if it works
max_proj_1 = max(vol_1, [], 3);
max_proj_2 = max(vol_2, [], 3);

max_vol_1 = [];
max_vol_2 = [];

for i = 1:size(vol_1, 3)
    max_vol_1 = cat(3, max_vol_1, max_proj_1);
    max_vol_2 = cat(3, max_vol_2, max_proj_2);
end

%% Define 3d partial kernels
dx = zeros(2,2,2); dx(:,:,1)=[-1 1; -1 1 ]; dx(:,:,2)=[-1 1; -1 1];
dy = zeros(2,2,2); dy(:,:,1)=[-1 -1; 1 1 ]; dy(:,:,2)=[-1 -1; 1 1];
dz = zeros(2,2,2); dz(:,:,1)=[-1 -1; -1 -1]; dz(:,:,2)=[1 1; 1 1];
dt = ones(2,2,2);
d_gain = 1;
dx = d_gain*dx; dy = d_gain*dy; dz = d_gain*dz; dt = d_gain*dt;

%% Define parameters
num_levels = 3;
window = 6;
win_size = floor(window/2);

%% pyramids creation
vol_1_pyramids{1} = vol_1;
vol_2_pyramids{1} = vol_2;
% vol_1_pyramids{1} = max_vol_1;
% vol_2_pyramids{1} = max_vol_2;
for i = 2:num_levels
    vol_1_pyramids{i} = impyramid_3d(vol_1_pyramids{i-1});
    vol_2_pyramids{i} = impyramid_3d(vol_1_pyramids{i-1});
end

%% Process all levels
for p = 1:num_levels
    %current pyramid, start from top
    vol_1 = vol_1_pyramids{num_levels - p + 1};
    vol_2 = vol_2_pyramids{num_levels - p + 1};
       
    if p == 1 % initialize
        u = zeros(size(vol_1));
        v = zeros(size(vol_1));
        w = zeros(size(vol_1));
    else %resizing
        u = 2 * imresizen(u, 2);   
        v = 2 * imresizen(v, 2);
        w = 2 * imresizen(w, 2);
    end
    
    % Lucas kanade refinement
    u = round(u); v = round(v); w = round(w);

    % Loop for every pixel
    for i = 1 + win_size:size(vol_1,1) - win_size 
        for j = 1 + win_size:size(vol_1,2) - win_size
            for k = 1 + win_size:size(vol_1,3) - win_size
                
                % Define boundaries
                r_2_low_idx = i - win_size + v(i,j,k);
                r_2_up_idx = i + win_size + v(i,j,k);
                c_2_low_idx = j - win_size + u(i,j,k);
                c_2_up_idx = j + win_size + u(i,j,k);
                d_2_low_idx = k - win_size + w(i,j,k);
                d_2_up_idx = k + win_size + w(i,j,k);

                % Handle boundary conditions
                if (r_2_low_idx < 1)...
                    ||(r_2_up_idx > size(vol_1,1))...
                    || (c_2_low_idx < 1) ...
                    || (c_2_up_idx > size(vol_1,2))...
                    || (d_2_low_idx < 1) ...
                    || (d_2_up_idx > size(vol_1,3))
                   continue; 
                end

                % Reference window
                r_1_patch_idx = i-win_size:i+win_size;
                c_1_patch_idx = j-win_size:j+win_size;
                d_1_patch_idx = k-win_size:k+win_size;
                window_1 = vol_1(r_1_patch_idx, c_1_patch_idx, d_1_patch_idx);
                % window_1(isnan(window_1)) = 0;
                % Moved window
                r_2_patch_idx = r_2_low_idx:r_2_up_idx;
                c_2_patch_idx = c_2_low_idx:c_2_up_idx;
                d_2_patch_idx = d_2_low_idx:d_2_up_idx;
                window_2 = vol_2(r_2_patch_idx, c_2_patch_idx, d_2_patch_idx);
                % window_2(isnan(window_2)) = 0;
                
                fx = convn(window_1, dx) + convn(window_2, dx);
                fy = convn(window_1, dy) + convn(window_2, dy);
                fz = convn(window_1, dz) + convn(window_2, dz);
                ft = convn(window_1, dt) + convn(window_2, -dt);

                idx_permutation = [1 2 3];
                Fx = fx(2:window-1,2:window-1,2:window-1);
                Fx = permute(Fx, idx_permutation);
                Fy = fy(2:window-1,2:window-1,2:window-1);
                Fy = permute(Fy, idx_permutation);
                Fz = fz(2:window-1,2:window-1,2:window-1);
                Fz = permute(Fz, idx_permutation);
                Ft = ft(2:window-1,2:window-1,2:window-1);
                Ft = permute(Ft, idx_permutation);
        
                A = [Fx(:) Fy(:) Fz(:)];
                b = -Ft(:);
                s = pinv(A)*b;

                u(i,j,k) = u(i,j,k) + s(1); 
                v(i,j,k) = v(i,j,k) + s(2);
                w(i,j,k) = w(i,j,k) + s(3);
        
            end
        end
    end
end


%% resizing
u = u(window:size(u,1)-window+1,window:size(u,2)-window+1,window:size(u,3)-window+1);
v = v(window:size(v,1)-window+1,window:size(v,2)-window+1,window:size(v,3)-window+1);
w = w(window:size(w,1)-window+1,window:size(w,2)-window+1,window:size(w,3)-window+1);

%% Pad to original image size
x_pad = round((size(vol_1, 1) - size(u, 1))/2);
y_pad = round((size(vol_1, 2) - size(u, 2))/2);
z_pad = round((size(vol_1, 3) - size(u, 3))/2);
u = padarray(u, [x_pad y_pad z_pad]);
v = padarray(v, [x_pad y_pad z_pad]);
w = padarray(w, [x_pad y_pad z_pad]);

%% colormap display
figure(1); clf;
max_u = max(u, [], 3);
max_v = max(v, [], 3);
RGB1=showmap3(max_u,max_v,5);
imagesc(RGB1);