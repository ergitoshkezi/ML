% Load the data from the .mat file
data = load('icecream.mat');

% Extract the variables from the loaded data
icecream2010 = data.icecream2010;  % Ice cream sales data for 2010
icecream2011 = data.icecream2011;  % Ice cream sales data for 2011
icecream2012 = data.icecream2012;  % Ice cream sales data for 2012
icecream2013 = data.icecream2013;  % Ice cream sales data for 2013
icecream2014 = data.icecream2014;  % Ice cream sales data for 2014
icecream2015 = data.icecream2015;  % Ice cream sales data for 2015
icecream2016 = data.icecream2016;  % Ice cream sales data for 2016
icecream2017 = data.icecream2017;  % Ice cream sales data for 2017
icecream2018 = data.icecream2018;  % Ice cream sales data for 2018
icecream2019 = data.icecream2019;  % Ice cream sales data for 2019
icecream2020 = data.icecream2020;  % Ice cream sales data for 2020
icecream2021 = data.icecream2021;  % Ice cream sales data for 2021

rain2010 = data.rain2010;  % Rain data for 2010
rain2011 = data.rain2011;  % Rain data for 2011
rain2012 = data.rain2012;  % Rain data for 2012
rain2013 = data.rain2013;  % Rain data for 2013
rain2014 = data.rain2014;  % Rain data for 2014
rain2015 = data.rain2015;  % Rain data for 2015
rain2016 = data.rain2016;  % Rain data for 2016
rain2017 = data.rain2017;  % Rain data for 2017
rain2018 = data.rain2018;  % Rain data for 2018
rain2019 = data.rain2019;  % Rain data for 2019
rain2020 = data.rain2020;  % Rain data for 2020
rain2021 = data.rain2021;  % Rain data for 2021
rain2022 = data.rain2022;  % Rain data for 2022

temperature2010 = data.temperature2010;  % Temperature data for 2010
temperature2011 = data.temperature2011;  % Temperature data for 2011
temperature2012 = data.temperature2012;  % Temperature data for 2012
temperature2013 = data.temperature2013;  % Temperature data for 2013
temperature2014 = data.temperature2014;  % Temperature data for 2014
temperature2015 = data.temperature2015;  % Temperature data for 2015
temperature2016 = data.temperature2016;  % Temperature data for 2016
temperature2017 = data.temperature2017;  % Temperature data for 2017
temperature2018 = data.temperature2018;  % Temperature data for 2018
temperature2019 = data.temperature2019;  % Temperature data for 2019
temperature2020 = data.temperature2020;  % Temperature data for 2020
temperature2021 = data.temperature2021;  % Temperature data for 2021
temperature2022 = data.temperature2022;  % Temperature data for 2022

% Insert NaNs for missing data points
% Ice cream sales data
icecream2010 = [icecream2010, NaN];   % Insert NaN for 2010
icecream2011 = [icecream2011, NaN];   % Insert NaN for 2011
icecream2013 = [icecream2013, NaN];   % Insert NaN for 2013
icecream2014 = [icecream2014, NaN];   % Insert NaN for 2014
icecream2015 = [icecream2015, NaN];   % Insert NaN for 2015
icecream2017 = [icecream2017, NaN];   % Insert NaN for 2017
icecream2018 = [icecream2018, NaN];   % Insert NaN for 2018
icecream2019 = [icecream2019, NaN];   % Insert NaN for 2019
icecream2021 = [icecream2021, NaN];   % Insert NaN for 2021

% Rain data
rain2010 = [rain2010, NaN];   % Insert NaN for 2010
rain2011 = [rain2011, NaN];   % Insert NaN for 2011
rain2013 = [rain2013, NaN];   % Insert NaN for 2013
rain2014 = [rain2014, NaN];   % Insert NaN for 2014
rain2015 = [rain2015, NaN];   % Insert NaN for 2015
rain2017 = [rain2017, NaN];   % Insert NaN for 2017
rain2018 = [rain2018, NaN];   % Insert NaN for 2018
rain2019 = [rain2019, NaN];   % Insert NaN for 2019
rain2021 = [rain2021, NaN];   % Insert NaN for 2021
rain2022 = [rain2022, NaN];   % Insert NaN for 2022

% Temperature data
temperature2010 = [temperature2010, NaN];   % Insert NaN for 2010
temperature2011 = [temperature2011, NaN];   % Insert NaN for 2011
temperature2013 = [temperature2013, NaN];   % Insert NaN for 2013
temperature2014 = [temperature2014, NaN];   % Insert NaN for 2014
temperature2015 = [temperature2015, NaN];   % Insert NaN for 2015
temperature2017 = [temperature2017, NaN];   % Insert NaN for 2017
temperature2018 = [temperature2018, NaN];   % Insert NaN for 2018
temperature2019 = [temperature2019, NaN];   % Insert NaN for 2019
temperature2021 = [temperature2021, NaN];   % Insert NaN for 2021
temperature2022 = [temperature2022, NaN];   % Insert NaN for 2022

% Combine data into features and targets
years = 2010:2021;

% Combine ice cream sales data for all years
icecream_data = [icecream2010; icecream2011; icecream2012; icecream2013; icecream2014;
                 icecream2015; icecream2016; icecream2017; icecream2018; icecream2019;
                 icecream2020; icecream2021];

% Combine rain data for all years
rain_data = [rain2010; rain2011; rain2012; rain2013; rain2014;
             rain2015; rain2016; rain2017; rain2018; rain2019;
             rain2020; rain2021];

% Combine temperature data for all years
temperature_data = [temperature2010; temperature2011; temperature2012; temperature2013; temperature2014;
                    temperature2015; temperature2016; temperature2017; temperature2018; temperature2019;
                    temperature2020; temperature2021];

% Calculate the seasonal component
% Compute the average sales for each day of the year across all years
seasonal_component = mean(reshape(icecream_data, 366, []), 2);

% Deseasonalize the ice cream sales data
deseasonalized_icecream_data = icecream_data - repmat(seasonal_component, 1, length(years))';

% Reshape data for regression models
% Each row in X will represent a sample with columns [year, rain, temperature]
X = [repmat(years', 366, 1), reshape(rain_data', [], 1), reshape(temperature_data', [], 1)];

% Reshape y to be a column vector of deseasonalized ice cream sales data across all years
y = reshape(deseasonalized_icecream_data', [], 1);

% Prepare the test set for 2022
% X_test will have the same structure as X, but for the year 2022
X_test = [repmat(2022, 366, 1), reshape(rain2022, [], 1), reshape(temperature2022, [], 1)];

% Feature Engineering: Add interaction terms
% Add an interaction term (rain * temperature) to the features
X = [X, X(:,2) .* X(:,3)];  % Add interaction term (rain * temperature) to training data
X_test = [X_test, X_test(:,2) .* X_test(:,3)];  % Add interaction term to test data

% Define models to train
models = {'Linear Regression', 'SVM', 'Random Forest', 'Neural Network'};

% Train models and make predictions for 2022
for i = 1:length(models)
    model = models{i};
    
    switch model
        case 'Linear Regression'
            mdl = fitlm(X, y);
        case 'SVM'
            mdl = fitrsvm(X, y, 'KernelFunction', 'rbf', 'Standardize', true);
        case 'Random Forest'
            mdl = TreeBagger(50, X, y, 'Method', 'regression');
        case 'Neural Network'
            % For simplicity, we will use a feedforward network with 10 hidden neurons.
            
            net = feedforwardnet(10);
            net = train(net, X', y');
            mdl = net;
    end
    
    % Make predictions for 2022
    if strcmp(model, 'Neural Network')
        y_pred = net(X_test')';
    else
        y_pred = predict(mdl, X_test);
    end
    
    % Add the seasonal component back to the predictions
    y_pred = y_pred + seasonal_component;
    
    % Print the predictions
    fprintf('%s: Predictions for 2022 = \n', model);
    disp(y_pred);
end
