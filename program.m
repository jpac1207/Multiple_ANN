% ---------- Parâmetros Gerais ----------
maxEpochs = 300; % Número de épocas do treinamento
H = 15; % Número de neurônios na camada escondida
I = 30; % Número de neurônios na camada de entrada
O = 6; % Número de neurônios na camada de saída
eta = 0.05; % Learning Rate utilizado no cálculo do backpropagation.

neuralNetworks(1).I = 30;
neuralNetworks(1).H = 15;
neuralNetworks(1).O = 6;
neuralNetworks(1).eta = 0.05;
neuralNetworks(1).numberOfEpochs = 300;
neuralNetworks(1).isTheFirstNetwork = 1;

neuralNetworks(2).I = 30;
neuralNetworks(2).H = 15;
neuralNetworks(2).O = 6;
neuralNetworks(2).eta = 0.05;
neuralNetworks(2).numberOfEpochs = 200;
neuralNetworks(2).isTheFirstNetwork = 0;

%doTraining(maxEpochs, I, H, O, eta, eta_gaussian)
doTraining2(neuralNetworks)

function doTraining2(neuralNetworks)
    processed_dataset = load('processed_dataset.mat');
    X = processed_dataset.X;
    Y = processed_dataset.Y;
    X_norm = normalizeInput(X);
    [X_train, Y_train, X_val, Y_val, X_test, Y_test] = splitData(X_norm, Y);

    % Compute the total number of epochs
    
    totalEpochs = 0;            
    for i=1:size(neuralNetworks, 2)        
        totalEpochs = totalEpochs + neuralNetworks(i).numberOfEpochs;
    end
    allNetworksErrors = zeros(totalEpochs, 1);
    allNetworksValErrors = zeros(totalEpochs, 1);
    start = 1;

    for i=1:size(neuralNetworks, 2)        
        [hiddenVsInputWeights, hiddenVsInputBias, outputVsHiddenWeights, ...
            outputVsHiddenBias, finalErrors, finalValErrors, ...
            trainingFinalPredictions, validationFinalPredictions] = trainMLP(neuralNetworks(i).I, ...
            neuralNetworks(i).H, neuralNetworks(i).O, neuralNetworks(i).numberOfEpochs, ...,
            neuralNetworks(i).eta, 0, X_train', Y_train, X_val', Y_val, neuralNetworks(i).isTheFirstNetwork);     
        allNetworksErrors(start:(start+neuralNetworks(i).numberOfEpochs-1), 1) = finalErrors;
        allNetworksValErrors(start:(start+neuralNetworks(i).numberOfEpochs-1), 1) = finalValErrors;
        Y_train = (Y_train .* (1-trainingFinalPredictions));
        Y_val = (Y_val .* (1-validationFinalPredictions));        
        start = start + neuralNetworks(i).numberOfEpochs;
        %trainingFinalPredictions
    end  
    plot(1:totalEpochs, allNetworksErrors, 'o');
    hold on;
    plot(1:totalEpochs, allNetworksValErrors, 'x');
    hold off; 
    legend('Média Erros Treinamento MLP', 'Média Erros Validação MLP');
    ylabel('Erro Quadrático Médio');
    xlabel('Épocas');
    title('Erros de Treino e Validação do Treinamento');  
end

function doTraining(maxEpochs, I, H, O, eta, eta_gaussian)
    processed_dataset = load('processed_dataset.mat');
    X = processed_dataset.X;
    Y = processed_dataset.Y;
    X_norm = normalizeInput(X);
    [X_train, Y_train, X_val, Y_val, X_test, Y_test] = splitData(X_norm, Y);
    allNetworksErrors = zeros(maxEpochs * 2, 1);
    allNetworksValErrors = zeros(maxEpochs * 2, 1);
%     [hiddenVsInputWeights, outputVsHiddenWeights, outputVsHiddenBias, sigmas, rbfFinalErrors, rbfFinalValErrors]  = trainRBF(I, H, O, maxEpochs, eta, eta_gaussian, ...
%            X_train', Y_train, X_val', Y_val)
    start = 1
    [hiddenVsInputWeights, hiddenVsInputBias, outputVsHiddenWeights, outputVsHiddenBias, finalErrors, finalValErrors, trainingFinalPredictions, validationFinalPredictions] = trainMLP(I, H, O, maxEpochs, eta, ...
        1, X_train', Y_train, X_val', Y_val, 0);     
    errorPerInstance = (Y_train .* (1-trainingFinalPredictions));
    validationErrorPerInstance = (Y_val .* (1-validationFinalPredictions));
    allNetworksErrors(start:(start+maxEpochs-1), 1) = finalErrors;
    allNetworksValErrors(start:(start+maxEpochs-1), 1) = finalValErrors;
    start = start + maxEpochs;
    trainingFinalPredictions
    errorPerInstance
    disp('---------------------')
    [hiddenVsInputWeights, hiddenVsInputBias, outputVsHiddenWeights, outputVsHiddenBias, finalErrors, finalValErrors, trainingFinalPredictions, validationFinalPredictions] = trainMLP(I, H * 2, O, maxEpochs, eta * 0.1, ...
        0, X_train', errorPerInstance, X_val', validationErrorPerInstance, 1);   
    allNetworksErrors(start:(start+maxEpochs-1), 1) = finalErrors;
    allNetworksValErrors(start:(start+maxEpochs-1), 1) = finalValErrors;
    start = start + maxEpochs;

    trainingFinalPredictions    

    plot((1:maxEpochs*2), allNetworksErrors, 'o');
    hold on;
    plot((1:maxEpochs*2), allNetworksValErrors, 'x');
    hold off; 
    legend('Média Erros Treinamento MLP', 'Média Erros Validação MLP');
    ylabel('Erro Quadrático Médio');
    xlabel('Épocas');
    title('Erros de Treino e Validação do Treinamento');
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