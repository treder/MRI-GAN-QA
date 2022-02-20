% analyze images using NIQE and BRISQUE
clear 
% modality = 'GM';
modality = 'T1';

basedir = '/Users/matthiastreder/git/MRI-GAN-QA/';
datadir = [basedir 'experiment/'];
imagedir = [basedir '3DStyleGAN/' modality '_png/'];

figdir = [basedir 'figures/'];
resultdir = [basedir '3DStyleGAN/results/'];

iteration = {'008','016','032','064','128','200','real'};

%% load StyleGAN images
images = cell(numel(iteration),1);

n_style_images = 100;

if strcmp(modality,'T1')
    tmp = uint8(zeros(n_style_images, 80, 240, 3));
else
    tmp = uint8(zeros(n_style_images, 288, 432, 3));
end

for ix = 1:numel(iteration)
    files = dir([imagedir iteration{ix} '/*.png']);
    files = sort({files.name});
    disp(iteration{ix})
    for iy = 1:n_style_images
        im = imread([imagedir iteration{ix} '/' files{iy}]);
        disp(files{iy})
        tmp(iy, :, :, :) = repmat(im(:,:,1), [1 1 3]);
    end
    images{ix} = tmp;
end

%% NIQE
df = table([],[],[],[],[]);
df.Properties.VariableNames = {'iteration' 'niqe' 'brisque' 'niqe-mri','brisque-mri (transfer)'};
% uses default blocksize of 96x96
for it = 1:numel(iteration)
    ims = images{it};
    for ix = 1:size(ims,1)
        score = niqe(squeeze(ims(ix, :,:,:)));
        df = [df; {iteration{it} score 0 0 0}];
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

imds = imageDatastore([imagedir 'for_brisque/'],'FileExtensions',{'.png'});
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

%% BRISQUE using custom model -> requires MOS which are not available for the StyleGAN images
% therefore we train on the Cam-CAN data and validate on the StyleGAN
n_rating_images = 30;
test_images_rating = uint8(zeros(n_rating_images, 288, 432, 3));
rating_filenames = cell(n_rating_images, 1);

df_rating = readtable([basedir 'experiment/Psytoolkit/rating_table.txt'], 'Format', '%s %d');
df_rating.Properties.VariableNames = {'image','batch'};
df_rating(df_rating{:,'batch'}>0,:);  % discard real images

for ix = 1:n_rating_images
    imfile = [basedir 'experiment/Psytoolkit/images/' df_rating{ix, 'image'}{1} '.png'];
    rating_filenames{ix} = imfile;
    test_images_rating(ix, :, :, :) = imread(imfile);
end

% Load human ratings
df_human = readtable([basedir 'results/psytoolkit_all_participants26.csv'],'format','auto');
df_human = df_human(strcmp(df_human{:,'task'},'RATING_TASK'), :);

% Apply high/low RT cutoff (same cutoff as in Python script)
df_human = df_human(df_human{:,'rate_RT'} > 150, :);
df_human = df_human(df_human{:,'rate_RT'} < 10000, :);

% recode batch: 344:0, 1055:1, 7954:2, 24440:3, 60000:4, 'real':5
for ix = 1:size(df_human, 1)
    b = df_human{ix, 'batch'}{1};
    switch(b)
        case '344', bcode = 0;
        case '1055', bcode = 1;
        case '7954', bcode = 2;
        case '24440', bcode = 3;
        case '60000', bcode = 4;
        case 'real', bcode = 5;
    end
    df_human{ix, 'batch_code'} = bcode;
end

% reduce cols
df_human = df_human(:, {'tablerow' 'rate_RT' 'rate' 'batch_code'});

% calculate table with Mean Opinion Scores (averages of rate)
MOS = grpstats(df_human, 'tablerow');
% MOS = MOS{:, 'mean_rate'} * 20; % for brisque MOS needs to span from 1 to 100 

% Differential MOS (DMOS): average MOS(real) - MOS(batch)
MOS_real = mean(MOS{MOS{:,'mean_batch_code'}==5,'mean_rate'});
DMOS = MOS_real - MOS{:,'mean_rate'} + 1; % include 'real' (+1 to make sure it's positive)
% DMOS = MOS_real - MOS{MOS{:,'mean_batch_code'}<5,'mean_rate'}; % exclude 'real'

% fit BRISQUE to MOS 
imds = imageDatastore(rating_filenames);
model = fitbrisque(imds, DMOS);

% test images must be (288, 432, 3) to match the fitted model, we thus have
% to pad for the T1 images
if strcmp(modality,'T1')
    for step = 1:numel(iteration)
        tmp = uint8(zeros(100, 288, 432, 3));
        tmp(:,105:end-104,97:end-96,:) = images{step};
        images{step} = tmp;
    end
end

% sanity check: scores on training data
% for ix = 1:n_rating_images
%     im = squeeze(test_images_rating(ix, :, :, :));
%     score = brisque(squeeze(im), model);
% end

% test on StyleGAN images
row = 1;
for it = 1:numel(iteration)
    ims = images{it};
    for ix = 1:size(ims,1)
        score = brisque(squeeze(ims(ix, :,:,:)), model);
        df{row, 'brisque-mri (transfer)'} = score;
        row = row + 1;
    end
end

%% save as CSV
writetable(df, [resultdir 'analyze_2D_image_NIQE_BRISQUE_' modality '.csv'])
