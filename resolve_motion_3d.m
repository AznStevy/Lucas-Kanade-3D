function motion = resolve_motion_3d(dataset_name, marker_root_name, ...
    marker_folderpath, bw_cell_folderpath, start_end, slices)

    %% Loop through volumes
    max_projections = [];
    all_centroids = {};
    op_flow = {};
    window_size = 50;

    for vol_idx = 1:length(start_end)
        vol_num = start_end(vol_idx);

        %% Load volume
        % If we want two volumes, so if i > 1, we assign a previous volume
        if vol_idx > 1
            prev_volume = volume; 
        end

        % Load new volume
        disp(strcat('Loading volume:  ', num2str(vol_num)));
        volume_name = strcat(marker_folderpath, '\', ...
            marker_root_name, num2str(vol_num), '.tif');
        volume = load_volume(volume_name, slices);
        
        % Load bw volume (non-interpolated)
%         disp(strcat('Loading bw cell volume:  ', num2str(vol_num)));
%         bw_volume_name = strcat(bw_cell_folderpath, '\', ...
%             marker_root_name, num2str(vol_num), '.tif');
%         bw_volume = load_volume(bw_volume_name);

        %% Create max projection for figure
        max_projections = cat(3, max_projections, max(volume, [], 3));

        %% Find good cells to track
        disp(strcat('Finding centroids for volume: ', ...
            num2str(vol_num)));

        [centroids, bw] = good_cells_to_track(volume, 10);
%         centroids = find_cell_centroids(bw_volume);
        all_centroids{vol_idx} = centroids;

        %% Set up cell states for optical flow
        % initialize cell states
        if vol_idx == 1
            cell_states = all_centroids{vol_idx};
        else % Define new cell state
            disp(strcat('Resolving motion for volume: ', ...
                num2str(start_end(vol_idx-1)),...
                '->',num2str(start_end(vol_idx))));
            cell_states = cat(3, cell_states, cell_states(:,:,vol_idx-1));

            %% Op flow (a priori/pre-optimization)
            % centroids because states might disappear
            features = all_centroids{vol_idx-1};
            [C, u, v, w] = optical_flow(...
                prev_volume, volume, features);
            op_flow{vol_idx-1}.C = C;
            op_flow{vol_idx-1}.u = u; 
            op_flow{vol_idx-1}.v = v; 
            op_flow{vol_idx-1}.w = w;
            op_flow_all_cells = [...
                op_flow{vol_idx-1}.C(:,1) + op_flow{vol_idx-1}.u, ...
                op_flow{vol_idx-1}.C(:,2) + op_flow{vol_idx-1}.v,...
                op_flow{vol_idx-1}.C(:,3) + op_flow{vol_idx-1}.w];

            for cell_num = 1:size(cell_states, 1)
                try
                    current_cell_pos = cell_states(cell_num, :, vol_idx-1);

                    % find correct optical flow result, closest
                    [q, op_cell_num] = ismember(current_cell_pos, ...
                        op_flow{vol_idx-1}.C, 'rows');

                    op_flow_cell_est = op_flow_all_cells(op_cell_num, :);

                    % assign the proper centroid
                    cell_states(cell_num, :, vol_idx) = op_flow_cell_est;
                catch
                    % leave it as it was.
                end
            end

            %% Assigning cells via a posteriori optimization
            % Define possible next centroids
            next_centroids = all_centroids{vol_idx}; % get next centroids
            costs = likelihoods_c_elegan(...
                cell_states, next_centroids, window_size);
            
            % Here, we already have an initial guess of the cell_state of i 
            % for the volume i from optical flow. Now we just need to 
            % optimize it.
            cell_states = optimize_c_elegan(...
                costs, cell_states, next_centroids, window_size);
        end
    end

    %% Resolve final motion
    resolved_motion = {};
    for vol_idx = 1:size(cell_states, 3)-1 
        resolved_motion{vol_idx}.C = cell_states(:, :, vol_idx);
        dP = cell_states(:, :, vol_idx+1) - cell_states(:, :, vol_idx);
        resolved_motion{vol_idx}.u = dP(:, 1);
        resolved_motion{vol_idx}.v = dP(:, 2);
        resolved_motion{vol_idx}.w = dP(:, 3);
    end

    %% Define cells to highlight
    % highlight_cells = [...
    %     2,3,6,13,20,25,26,29,31,32,33,...
    %     34,36,37,38,40,41,42,44,45,51,...
    %     ]; 
    highlight_cells = 1:1:size(cell_states, 1);

    %% save resolved motion
    motion.op_flow = op_flow;
    motion.resolved_motion = resolved_motion;
    motion.cell_states = cell_states;
    motion.max_projections = max_projections;
    motion.all_centroids = all_centroids;
    motion.volumes = start_end;
    motion.vol_dim = size(prev_volume);
    motion.slices = slices;
    motion.window_size = window_size;
    motion.plot.slice_xy_space = Inf;
    motion.plot.slice_z_space = Inf;
    motion.plot.highlighted_cells = highlight_cells;

    save_name = strcat('motion_data/motion_',dataset_name,'_', ...
        num2str(start_end(1)), '_', num2str(start_end(end)));
    motion.dataset_name = dataset_name;
    motion.save_name = save_name;

    mkdir('motion_data');
    save(save_name, 'motion')
end