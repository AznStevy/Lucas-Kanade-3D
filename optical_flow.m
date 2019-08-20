function [C, u_cen, v_cen, w_cen] = optical_flow(...
    vol_1, vol_2, features)
%% Load volumes
% vol_1 = load_volume('data_2/SPIMA-0.tif');
% vol_2 = load_volume('data_2/SPIMA-1.tif');

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
for i = 2:num_levels
    vol_1_pyramids{i} = impyramid_3d(vol_1_pyramids{i-1});
    vol_2_pyramids{i} = impyramid_3d(vol_1_pyramids{i-1});
end

%% Process all levels
for p = 1:num_levels
    % define multiplier
    level_scale = 2^(num_levels - p);
    
    % current pyramid, start from top
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

    % Loop for every pixel for a level
    for i = 1 + win_size:size(vol_1,1) - win_size 
        for j = 1 + win_size:size(vol_1,2) - win_size
            for k = 1 + win_size:size(vol_1,3) - win_size
                
                % if it's close by to any features.
                scaled_i_range = round([i-win_size-2, i+win_size+2]*level_scale);
                scaled_j_range = round([j-win_size-2, j+win_size+2]*level_scale);
                scaled_k_range = round([k-win_size-2, k+win_size+2]*level_scale);
                
                % determine if any features are within the range
                i_in_range = scaled_i_range(1) <= features(:, 1) & ...
                    features(:, 1) <= scaled_i_range(2);
                j_in_range = scaled_j_range(1) <= features(:, 2) & ...
                    features(:, 2) <= scaled_j_range(2);
                k_in_range = scaled_k_range(1) <= features(:, 3) & ...
                    features(:, 3) <= scaled_k_range(2);
                in_range = i_in_range .* j_in_range .* k_in_range;
                
                % only do this if it's within range of a centroid
                if ~any(in_range); continue; end;

                % Define boundaries
                r_2_low_idx = i - win_size + v(i,j,k);
                r_2_up_idx  = i + win_size + v(i,j,k);
                c_2_low_idx = j - win_size + u(i,j,k);
                c_2_up_idx  = j + win_size + u(i,j,k);
                d_2_low_idx = k - win_size + w(i,j,k);
                d_2_up_idx  = k + win_size + w(i,j,k);

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
                % Moved window
                r_2_patch_idx = r_2_low_idx:r_2_up_idx;
                c_2_patch_idx = c_2_low_idx:c_2_up_idx;
                d_2_patch_idx = d_2_low_idx:d_2_up_idx;
                window_2 = vol_2(r_2_patch_idx, c_2_patch_idx, d_2_patch_idx);

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

%% For each point, determine C, u, v, w
C = features;
u_cen = []; v_cen = []; w_cen = [];
dir_window = 6;
for i = 1:size(features, 1)
    % centroid position
    curr_centroid = round(features(i, :));
    % find average of all non-zero elements within a window
    r_idx = max(curr_centroid(1)-dir_window, 0):min(curr_centroid(1)+dir_window, size(u, 1));
    c_idx = max(curr_centroid(2)-dir_window, 0):min(curr_centroid(2)+dir_window, size(u, 2));
    d_idx = max(curr_centroid(3)-dir_window, 0):min(curr_centroid(3)+dir_window, size(u, 3));
    % r_idx = 2*r_idx; c_idx = 2*c_idx; d_idx = 2*d_idx;
    
    u_window = u(r_idx, c_idx, d_idx); u_window(isnan(u_window))=0;
    v_window = v(r_idx, c_idx, d_idx); v_window(isnan(v_window))=0;
    w_window = w(r_idx, c_idx, d_idx); w_window(isnan(w_window))=0;
    u_avg = mean(nonzeros(u_window(:))); u_avg(isnan(u_avg)) = 0;
    v_avg = mean(nonzeros(v_window(:))); v_avg(isnan(v_avg)) = 0;
    w_avg = mean(nonzeros(w_window(:))); w_avg(isnan(w_avg)) = 0;
    u_cen = [u_cen u_avg];
    v_cen = [v_cen v_avg];
    w_cen = [w_cen w_avg];
end

%% display
% figure(1); clf;
% rows = 1; cols = 3;
% max_u = max(u, [], 3);
% max_v = max(v, [], 3);
% dis_u = permute(u, [2 1 3]);
% dis_v = permute(v, [2 1 3]);
% dis_w = permute(w, [2 1 3]);
% 
% hold on; grid on;
% [XX, YY, ZZ] = meshgrid(1:size(u,1), 1:size(u,2), 1:size(u,3));
% plot3(features(:, 1), features(:, 2), features(:, 3), ...
%     'o'); axis equal tight; 
% scale_factor = 1.0;
% quiver3(scale_factor*XX, scale_factor*YY, scale_factor*ZZ, ...
%     dis_u, dis_v, dis_w);
% hold off;
% xlim([1 size(vol_1, 2)]);
% ylim([1 size(vol_1, 1)]);
% zlim([1 size(vol_1, 3)]);