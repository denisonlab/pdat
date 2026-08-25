function p = uncertainty_eeg_params

p.eeg_file = 'cupcake_S0085_allcat_oref_nofilt.mat';
p.trial_file = 'cupcake_S0085_allcat.mat';
p.timewindow = [-200, 1200];

%what feature to decode (from expt.trialsPresented)
p.feature = 'theta';
%specify whether the feature is circular or not
p.iscircular = 1;
%decoding of non-circular features not currently supported by encoding
%model method, but is supported by SVR

%since SVM is typically used for categorical features, you can create n
%categorical bins for use with it here (or 0 for don't do this)
%don't forget to set regressionmethod to 'SVM'
p.featurebins = 0;
%bins have the same number of trials, but not necessarily uniformly spaced
%edges

%subsample trials such that the overall dataset has equal number of trials
%in each of n evenly spaced bins (set to number of bins, 0 for don't do this)
p.featurebin_subsamp = 4; %this is done totally separate and prior to the featurebins option above

%number of bagging iterations in decoding function. If this is set to 1,
%it's equivalent to not bagging - i.e. every unique trial is used exactly
%once
p.nboot = 10;
%leave-one-out cross-validation or p.test_part split cross-validation
p.loocv = 0;
%portion of data to hold out as testing data (I'm not looping over folds to
%get every trial but that could be added)
p.test_part = 0.2;
%what outputs to save in results.mat (postmat takes up a lot of memory)
p.dontsave = {'postmat'};

%currently does nothing I think
p.report = 0;

%ctrl-f to find what this does
p.debug = 0;

%which channels to use by index (may update to allow for channel names)
%leave empty for all channels
p.targetchannels = [];

%how to fit the weight matrix ('OLS', 'lasso', 'ridge', 'FLS' (flexible least
%squares, where weights change smoothly over time according to p.flsmu))
%or set this to "SVM" to use svm instead of the encoding model
p.regressionmethod = 'OLS';
p.flsmu = 1;

%'SVR' support vector regression (fitrlinear) or 'class' typical categorical SVM
p.svmmethod = 'SVR';

%'all', 'session 1-2', 'session 2-1', 'precue', 'nocue'
p.train_test = 'all';

%instead of the signal in the eeg file, take the power within a particular
%frequency range (0 for don't do this)
p.freq_range = 0;
%does it still make sense to do sliding window? pattern z-score?


%do we remove channels that were interpolated in preprocessing? (I think
%generally this is correct to do because we don't want linear dependence, 
%however if you have a concatenated dataset where different channels are
%interpolated in different trials it's complicated)
p.rmInterp = 0;


% THESE TWO OPTIONS ARE CURRENTLY INCOMPATIBLE WITH FEATURE_BIN_SUBSAMP ->
%subset training set into this portion to test validity with smaller trial
%numbers
%leave empty to not do this
p.train_subset = [];
%other way to control training set size, just put this here to get the
%exact number of trials, you just have to know it's gonna be an amount that
%works with the dataset size
p.ntraintrials = [];
% <- THESE TWO OPTIONS ARE CURRENTLY INCOMPATIBLE WITH FEATURE_BIN_SUBSAMP

%I have some preprocessed EEG files where excludetrials is a cell with each entry a
%different set of trials to exclude. this is the index or leave empty for
%no cell.
p.eyecutoff = [];


%number of basis functions to use for defining tuning curves
%this parameter must be less than the number of unique stimulus values
%presented across trials
%unless you compute weights separately
p.nchan = 8;
%normalize channel response (:= channel sum is equal for each stimulus
%value)
p.normchan = 1;
%number of sets of basis functions to choose from during bootstrapping
p.nsets = 1;
%exponent to which basis functions are raised
p.chan_exp = 5;
%number of bins to discretize possible stimulus values
p.nbinsstimval = 100;
%size of time bins
p.msperbin = 10;
%non-overlapping bins if 0, else index t will have info from (t,t+msperbin-1)
p.slidingwindow = 1;

%see fitcov function for details on these options
p.covmethod = 'diag_shrink';
p.lossmethod = 'numeric';
p.shrink_diag = 0;

%specific to covmethod TAFKAP_PCA, which is not currently supported
p.npcs_tafkap = [];
%specific to covmethod EM
p.FA_iters = 10000; %max number of iterations
p.FA_thresh = 10e-4; %stops EM loop if euclidean distance between previous cov and new cov < p.FA_thresh
%number of folds in cross-validation for covariance fitting
p.cvk = 4;
%relevant for various covmethods
p.gamma_range = linspace(0,1,500);
p.lambda_range = linspace(0,1,20);
p.lambda_var_range = linspace(0,1,100);
p.k_range = 1:20;
%ensure covariance is positive semi-definite by rounding negative or 0
%eigenvalues to 10^(-p.smooth_alpha)
p.smooth_cov = 1;
p.smooth_alpha = 10;

%regularization options in order that they are done
p.dobaseline = 0;
%number of ms before stimulus onset to average then subtract (each sensor
%each trial)
%for consistency with things like EEGlab, I should make this be like [-200, 0]
p.baselinetimewindow = 200;
p.dosensordemean = 0;
p.dosensorzscore = 1;
p.dopatternzscore = 0;

%average small numbers of trials in the same condition (not currently
%supported)
p.pseudotrials = 0;

%ways to get predictions and uncertainty from probability distribution
%'pop_vec', 'max'
p.outpmethod = 'pop_vec';
%'pop_vec', 'max', 'chansum', 'entropy'
p.uncmethod = 'pop_vec';

p.verbose = 1;

%do pca on electrode activity or no
p.pca = 0;
p.npcs = 50;
%crosstime = 1: temporal generalization, create an ntimepoints by
%ntimepoints matrix of results (not currently tested)
%crosstime = 0: always train and test on the same timepoint
p.crosstime = 0;

p.outputformat = 'single'; %has to be single usually

%shuffle trial labels to create a null distribution (results if there was no stimulus information in the EEG signal)
p.gennull = 0;

end