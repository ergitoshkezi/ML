

% Load the data from the .mat file
data = load('icecream.mat');

% Extract the variables from the loaded data
icecream2010 = data.icecream2010;
icecream2011 = data.icecream2011;
icecream2012 = data.icecream2012;
icecream2013 = data.icecream2013;
icecream2014 = data.icecream2014;
icecream2015 = data.icecream2015;
icecream2016 = data.icecream2016;
icecream2017 = data.icecream2017;
icecream2018 = data.icecream2018;
icecream2019 = data.icecream2019;
icecream2020 = data.icecream2020;
icecream2021 = data.icecream2021;
rain2010 = data.rain2010;
rain2011 = data.rain2011;
rain2012 = data.rain2012;
rain2013 = data.rain2013;
rain2014 = data.rain2014;
rain2015 = data.rain2015;
rain2016 = data.rain2016;
rain2017 = data.rain2017;
rain2018 = data.rain2018;
rain2019 = data.rain2019;
rain2020 = data.rain2020;
rain2021 = data.rain2021;
rain2022 = data.rain2022;
temperature2010 = data.temperature2010;
temperature2011 = data.temperature2011;
temperature2012 = data.temperature2012;
temperature2013 = data.temperature2013;
temperature2014 = data.temperature2014;
temperature2015 = data.temperature2015;
temperature2016 = data.temperature2016;
temperature2017 = data.temperature2017;
temperature2018 = data.temperature2018;
temperature2019 = data.temperature2019;
temperature2020 = data.temperature2020;
temperature2021 = data.temperature2021;
temperature2022 = data.temperature2022;

icecream2010 = [icecream2010, NaN];
icecream2011 = [icecream2011, NaN];
icecream2013 = [icecream2013, NaN];
icecream2014 = [icecream2014, NaN];
icecream2015 = [icecream2015, NaN];
icecream2017 = [icecream2017, NaN];
icecream2018 = [icecream2018, NaN];
icecream2019 = [icecream2019, NaN];
icecream2021 = [icecream2021, NaN];

rain2010 = [rain2010, NaN];
rain2011 = [rain2011, NaN];
rain2013 = [rain2013, NaN];
rain2014 = [rain2014, NaN];
rain2015 = [rain2015, NaN];
rain2017 = [rain2017, NaN];
rain2018 = [rain2018, NaN];
rain2019 = [rain2019, NaN];
rain2021 = [rain2021, NaN];
rain2022 = [rain2022, NaN];

temperature2010 = [temperature2010, NaN];
temperature2011 = [temperature2011, NaN];
temperature2013 = [temperature2013, NaN];
temperature2014 = [temperature2014, NaN];
temperature2015 = [temperature2015, NaN];
temperature2017 = [temperature2017, NaN];
temperature2018 = [temperature2018, NaN];
temperature2019 = [temperature2019, NaN];
temperature2021 = [temperature2021, NaN];
temperature2022 = [temperature2022, NaN];

% ... (load and preprocess data)

% Combine data into features and targets
years = 2010:2021;
icecream_data = [icecream2010; icecream2011; icecream2012; icecream2013; icecream2014;
                 icecream2015; icecream2016; icecream2017; icecream2018; icecream2019;
                 icecream2020; icecream2021];
rain_data = [rain2010; rain2011; rain2012; rain2013; rain2014;
             rain2015; rain2016; rain2017; rain2018; rain2019;
             rain2020; rain2021];
temperature_data = [temperature2010; temperature2011; temperature2012; temperature2013; temperature2014;
                    temperature2015; temperature2016; temperature2017; temperature2018; temperature2019;
                    temperature2020; temperature2021];

% Reshape data for regression models
X = [repmat(years', 366, 1), reshape(rain_data', [], 1), reshape(temperature_data', [], 1)];
y = reshape(icecream_data', [], 1);

% Prepare the test set for 2022
X_test = [repmat(2022, 366, 1), reshape(rain2022, [], 1), reshape(temperature2022, [], 1)];

% Feature Engineering: Add interaction terms
X = [X, X(:,2).*X(:,3)];
X_test = [X_test, X_test(:,2).*X_test(:,3)];

% Define and train the models
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
    y_pred = predict(mdl, X_test);
    
    % Print the predictions
    fprintf('%s: Predictions for 2022 = \n', model);
    disp(y_pred);
end
