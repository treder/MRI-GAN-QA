% analyze images using NIQE and BRISQUE
clear 

basedir = '/Users/matthiastreder/git/MRI-GAN-QA/';
datadir = [basedir 'experiment/'];
imagedir = [basedir 'experiment/Psytoolkit/'];
figdir = [basedir 'figures/'];
resultdir = [basedir 'results/'];

batches = {'batch_344', 'batch_1055', 'batch_7954', 'batch_24440', 'batch_60000', 'real'};

%% read table with test images (either from the detection task or from the rating task)

df_detection = readtable([datadir 'Psytoolkit/main_experiment_table.txt']);
df_detection.Properties.VariableNames = {'image','button','batch'};
df_detection(1:3,:)

df_rating = readtable([datadir 'Psytoolkit/rating_table.txt'], 'Format', '%s %d');
df_rating.Properties.VariableNames = {'image','batch'};
df_rating

    
n_detection_images = size(df_detection,1);
n_rating_images = size(df_rating, 1);

%% Load test images (used in the behavioral experiment)
test_images_detection = uint8(zeros(n_detection_images, 135, 355, 3));
test_images_rating = uint8(zeros(n_rating_images, 135, 355, 3));
rating_filenames = cell(n_rating_images, 1);

for ix = 1:n_detection_images
    im = imread([imagedir 'images/' df_detection{ix, 'image'}{1} '.png']);
    im = im(76:210,46:400, :);  % cut away the empty area around the MRI
    test_images_detection(ix, :, :, :) = im;
end

for ix = 1:n_rating_images
    imfile = [imagedir 'images/' df_rating{ix, 'image'}{1} '.png'];
    im = imread(imfile);
    im = im(76:210,46:400, :);  % cut away the empty area around the MRI
    test_images_rating(ix, :, :, :) = im;
    rating_filenames{ix} = imfile;
end

%% NIQE
% uses default blocksize of 96x96
for ix = 1:n_detection_images
    score = niqe(squeeze(test_images_detection(ix, :,:,:)));
    df_detection{ix, 'niqe'} = score;
end

for ix = 1:n_rating_images
    score = niqe(squeeze(test_images_rating(ix, :,:,:)));
    df_rating{ix, 'niqe'} = score;
end

%% BRISQUE
for ix = 1:n_detection_images
    score = brisque(squeeze(test_images_detection(ix, :,:,:)));
    df_detection{ix, 'brisque'} = score;
end

for ix = 1:n_rating_images
    score = brisque(squeeze(test_images_rating(ix, :,:,:)));
    df_rating{ix, 'brisque'} = score;
end

%% NIQE using custom model

% pristine reference images
targetdir = [datadir 'images_excluding_images_used_in_experiment/real/'];
imds = imageDatastore(targetdir,'FileExtensions',{'.png'});

model = fitniqe(imds);
% model = fitniqe(imds, 'BlockSize', [32 32]);

for ix = 1:n_detection_images
    score = niqe(squeeze(test_images_detection(ix, :,:,:)), model);
    df_detection{ix, 'niqe-mri'} = score;
end

for ix = 1:n_rating_images
    score = niqe(squeeze(test_images_rating(ix, :,:,:)), model);
    df_rating{ix, 'niqe-mri'} = score;
end

%% BRISQUE using custom model
% for brisque model fitting we need the original images (without cutting
% away the empty part)
test_images_detection = uint8(zeros(n_detection_images, 288, 432, 3));
test_images_rating = uint8(zeros(n_rating_images, 288, 432, 3));

for ix = 1:n_detection_images
    im = imread([imagedir 'images/' df_detection{ix, 'image'}{1} '.png']);
    test_images_detection(ix, :, :, :) = im;
end

for ix = 1:n_rating_images
    imfile = [imagedir 'images/' df_rating{ix, 'image'}{1} '.png'];
    im = imread(imfile);
    test_images_rating(ix, :, :, :) = im;
end

% Load human ratings
df_human = readtable([resultdir 'psytoolkit_all_participants26.csv'],'format','auto');
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
clear MOS

% RATING task (we have to use cross-validation since the rating images are used
% to train the model too)
n_folds = 5;
CV = cvpartition(df_rating{:,'batch'},'KFold', n_folds); % include 'real'
% CV = cvpartition(df_rating{1:end-5,'batch'},'KFold', n_folds); % exclude 'real'
for f = 1:n_folds
    fprintf('Fold #%d\n', f)
    % fit BRISQUE model on train data
    train_ix = find(CV.training(f));
    imds = imageDatastore(rating_filenames(train_ix));
    model = fitbrisque(imds, DMOS(train_ix));
    
    % apply to test data (including real images)
    test_ix = find(CV.test(f));
%     test_ix = [find(CV.test(f));(26:30)'];
    for ix = 1:numel(test_ix)
        score = brisque(squeeze(test_images_rating(test_ix(ix), :,:,:)), model);
        disp(score)
        df_rating{test_ix(ix), 'brisque-mri'} = score;
    end

end

% DETECTION task
imds = imageDatastore(rating_filenames);
% imds = imageDatastore(rating_filenames(1:end-5));
model = fitbrisque(imds, DMOS);

for ix = 1:n_detection_images
    score = brisque(squeeze(test_images_detection(ix, :,:,:)), model);
    df_detection{ix, 'brisque-mri'} = score;
end

%% save as CSV
writetable(df_rating, [resultdir 'analyze_2D_image_NIQE_BRISQUE_rating.csv'])
writetable(df_detection, [resultdir 'analyze_2D_image_NIQE_BRISQUE_detection.csv'])
