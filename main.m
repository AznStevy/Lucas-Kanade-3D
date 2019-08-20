%% Main function for cell-tracking
clear all; close all;
warning('off','all')
addpath(genpath(pwd));

%% Read in calcium volume data
dataset_name = 'evan_seam_';
marker_folderpath = 'C:\Users\xusz\Documents\Software\new_optical_flow\data_2';
bw_cell_folderpath = '';
marker_root_name = 'SPIMA-';
calcium_folderpath = '';
calcium_root_name = '';
volume_nums = 10:37; % 0:598;
slices = 1:35;

%% Perform actions
motion = resolve_motion_3d(dataset_name, marker_root_name, ...
    marker_folderpath, bw_cell_folderpath, volume_nums, slices);
%% Display plots
display_motion_plots(motion);
