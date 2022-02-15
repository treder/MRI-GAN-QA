% like analyze_2D_image_NIQE_BRISQUE but here we ignore the task 
% and simply load 100 images for each batch
clear 

basedir = '/Users/matthiastreder/git/MRI-GAN-QA/';
imagedir = [basedir 'experiment/'];
figdir = [basedir 'figures/'];
resultdir = [basedir 'results/'];

iteration = {'batch_344', 'batch_1055', 'batch_7954', 'batch_24440', 'batch_60000', 'real'};

%% Load test images (used in the behavioral experiment)
n_images = 100;
images = cell(numel(iteration),1);

tmp = uint8(zeros(n_images, 135, 355, 3));

step = 1;
for it = iteration
    files = dir([imagedir it{1} '/*.png']);
    files = sort({files.name});
    for ix = 1:n_images
        im = imread([imagedir it{1} '/' files{ix}]);
        disp(files{ix})
        im = im(76:210,46:400, :);  % cut away the empty area around the MRI
        tmp(ix, :, :, :) = repmat(im(:,:,1), [1 1 3]);
    end
    images{step} = tmp;
    step = step + 1;
end

%% NIQE
df = table([],[],[],[]);
df.Properties.VariableNames = {'iteration' 'niqe' 'brisque' 'niqe-mri'};
% uses default blocksize of 96x96
for it = 1:numel(iteration)
    ims = images{it};
    for ix = 1:size(ims,1)
        score = niqe(squeeze(ims(ix, :,:,:)));
        df = [df; {iteration{it} score 0 0}];
    end
end

%% BRISQUE
row = 1;
for it = 1:numel(iteration)
    ims = images{it};
    for ix = 1:size(ims,1)
        score = brisque(squeeze(ims(ix, :,:,:)));
        df{row, 'brisque'} = score;
        row = row + 1;
    end
end

%% NIQE using custom model

imds = imageDatastore([imagedir 'notask_for_brisque/'],'FileExtensions',{'.png'});
model = fitniqe(imds);
% model = fitniqe(imds, 'BlockSize', [32 32]);

row = 1;
for it = 1:numel(iteration)
    ims = images{it};
    for ix = 1:size(ims,1)
        score = niqe(squeeze(ims(ix, :,:,:)), model);
        df{row, 'niqe-mri'} = score;
        row = row + 1;
    end
end


%% BRISQUE using custom model -> requires MOS which are not available for this selection of images

%% save as CSV
writetable(df, [resultdir 'analyze_2D_image_NIQE_BRISQUE_notask.csv'])
