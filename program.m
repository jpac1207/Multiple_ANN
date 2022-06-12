
neuralNetworks = getNeuralNetworks();
doTraining(neuralNetworks)

% Realiza o treinamento das redes neurais utilizando os dados já
% processados, além de salvar os pesos aprendidos.
function doTraining(neuralNetworks)
    processed_dataset = load('processed_dataset.mat');
    X = processed_dataset.X;    
    Y = processed_dataset.Y;
    X_norm = normalizeInput(X);
    neuralNetworksCount = size(neuralNetworks, 2);
    [X_train, Y_train, X_val, Y_val, X_test, Y_test] = splitData(X_norm, Y);
    inputSize  = size(X, 2);
    outputSize  = size(Y, 1);

    % Computa número total de epócas    
    totalEpochs = 0;            
    for i=1:neuralNetworksCount        
        totalEpochs = totalEpochs + neuralNetworks(i).numberOfEpochs;
    end
    allNetworksErrors = zeros(totalEpochs, 1);
    allNetworksValErrors = zeros(totalEpochs, 1);
    start = 1;
    
    % Salva informações básicas
    save("./ann_weigths/basic_info", "neuralNetworksCount", "inputSize", "outputSize");

    for i=1:neuralNetworksCount        
        [hiddenVsInputWeights, hiddenVsInputBias, outputVsHiddenWeights, ...
            outputVsHiddenBias, finalErrors, finalValErrors, ...
            trainingFinalPredictions, validationFinalPredictions] = trainMLP(neuralNetworks(i).I, ...
            neuralNetworks(i).H, neuralNetworks(i).O, neuralNetworks(i).numberOfEpochs, ...,
            neuralNetworks(i).eta, 0, X_train', Y_train, X_val', Y_val, neuralNetworks(i).isTheFirstNetwork);     
        allNetworksErrors(start:(start+neuralNetworks(i).numberOfEpochs-1), 1) = finalErrors;
        allNetworksValErrors(start:(start+neuralNetworks(i).numberOfEpochs-1), 1) = finalValErrors;
        Y_train = (Y_train .* (max(Y_train)-trainingFinalPredictions));
        Y_val = (Y_val .* (max(Y_val)-validationFinalPredictions));        
        start = start + neuralNetworks(i).numberOfEpochs;       
        %trainingFinalPredictions
        % Saving the weights
        save("./ann_weigths/ann_weights_" + i + ".mat", "hiddenVsInputWeights", "hiddenVsInputBias", "outputVsHiddenWeights", "outputVsHiddenBias");
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

% Realiza o preenchimento do array 'neuralNetworks' com os parâmetros de
% cada rede neural definidos como um atributo na respectiva struct
function[neuralNetworks] = getNeuralNetworks()
    neuralNetworks(1).I = 30;
    neuralNetworks(1).H = 10;
    neuralNetworks(1).O = 6;
    neuralNetworks(1).eta = 0.1;
    neuralNetworks(1).numberOfEpochs = 500;
    neuralNetworks(1).isTheFirstNetwork = 1;
    
    neuralNetworks(2).I = 30;
    neuralNetworks(2).H = 15;
    neuralNetworks(2).O = 6;
    neuralNetworks(2).eta = 0.05;
    neuralNetworks(2).numberOfEpochs = 200;
    neuralNetworks(2).isTheFirstNetwork = 0;

    neuralNetworks(3).I = 30;
    neuralNetworks(3).H = 25;
    neuralNetworks(3).O = 6;
    neuralNetworks(3).eta = 0.01;
    neuralNetworks(3).numberOfEpochs = 100;
    neuralNetworks(3).isTheFirstNetwork = 0;
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