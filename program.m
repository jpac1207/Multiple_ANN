% ---------- Parâmetros Gerais ----------
maxEpochs = 1000; % Número de épocas do treinamento
H = 25; % Número de neurônios na camada escondida
I = 30; % Número de neurônios na camada de entrada
O = 6; % Número de neurônios na camada de saída
eta = 0.05; % Learning Rate utilizado no cálculo do backpropagation.
eta_gaussian = 0.1; % Learning Rate utilizado no cálculo da atualização de centro dos neurônios de ativação gaussiana.

doTraining(maxEpochs, I, H, O, eta, eta_gaussian)

function doTraining(maxEpochs, I, H, O, eta, eta_gaussian)
    processed_dataset = load('processed_dataset.mat');
    X = processed_dataset.X;
    Y = processed_dataset.Y;
    X_norm = normalizeInput(X);
    [X_train, Y_train, X_val, Y_val, X_test, Y_test] = splitData(X_norm, Y);
    [hiddenVsInputWeights, outputVsHiddenWeights, outputVsHiddenBias, sigmas, finalErrors, finalValErrors]  = trainRBF(I, H, O, maxEpochs, eta, eta_gaussian, ...
           X_train', Y_train, X_val', Y_val)
    %[hiddenVsInputWeights, hiddenVsInputBias, outputVsHiddenWeights, outputVsHiddenBias, finalErrors, finalValErrors] = trainMLP(I, H, O, maxEpochs, eta, ...
    %    1, X_train', Y_train, X_val', Y_val)
    plot((1:maxEpochs), finalErrors, 'o');
    hold on;
    plot((1:maxEpochs), finalValErrors, 'x');
    hold off;
    legend('Média Erros Treinamento', 'Média Erros Validação');
end


% Realiza a divisão dos dados contidos em 'X' e 'Y' em:
% X_train -> Padrões de entrada a serem utilizados no treino (70%)
% Y_train -> Padrões de saída a serem utilizados no treino (70%)
% X_val -> Padrões de entrada a serem utilizados na validação (20%)
% Y_val -> Padrões de saída a serem utilizados na validação (20%)
% X_test -> Padrões de entrada a serem utilizados no teste (10%)
% Y_test -> Padrões de saída a serem utilizados no testw (10%)
function [X_train, Y_train, X_val, Y_val, X_test, Y_test] = splitData(X, Y)
    numberOfRows = size(X, 1);
    trainProportion = 0.7;
    trainRows = floor(numberOfRows * trainProportion);
    valProportion = 0.2;
    valRows = floor(numberOfRows * valProportion);
    testProportion = 0.1;
    testRows = floor(numberOfRows * testProportion);    

    randIndexes = randperm(numberOfRows);   
    trainIndexes = randIndexes(1:trainRows);    
    initOfValRows = (trainRows + 1);
    valIndexes = randIndexes(initOfValRows:(initOfValRows + valRows - 1));
    initOfTestRows = (initOfValRows + valRows);
    testIndexes = randIndexes(initOfTestRows:(initOfTestRows + testRows - 1));

    X_train = X(trainIndexes, :);
    Y_train = Y(:, trainIndexes);
    
    X_val = X(valIndexes, :);
    Y_val = Y(:, valIndexes);
    
    X_test = X(testIndexes, :);
    Y_test = Y(:, testIndexes);
end