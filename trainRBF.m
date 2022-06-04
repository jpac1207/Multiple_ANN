% Realiza o treinamento da RBF, de acordo com os parametros:
% I -> Número de neurônios na camada de entrada
% H -> Número de neurônios na camada escondida
% O -> Número de neurônios na camada de saída
% maxEpochs -> Número de epócas do treinamento
% eta -> Taxa de aprendizado
% activationType -> Flag utilizada para definir a função de ativação da
% camada escondida
% X_train -> Padrões de entrada utilizados durante o treinamento
% Y_train -> Padrões de saída utilizados durante o treinamento
% X_val -> Padrões de entrada utilizados na validação
% Y_val -> Padrões de saída utilizados na validação
function [hiddenVsInputWeights, outputVsHiddenWeights, outputVsHiddenBias, sigmas, finalErrors, finalValErrors] = trainRBF(I, H, O, maxEpochs, eta, ...
    eta_gaussian, X_train, Y_train, X_val, Y_val)
    currentEpoch = 1;    
    errors = zeros(maxEpochs, 1);  
    validationErrors = zeros(maxEpochs, 1);
    % Número de padrões de entrada
    numberOfTrainingInstances = size(X_train, 2);
    % Número de padrões de validação
    numberOfValidationInstances = size(X_val, 2);    
    % Centros camada escondida
    C = rand(H, I) - 0.5;     
    % Pesos entre camada escondida e camada de saída
    Woh = rand (O, H) - 0.5;
    % Bias entre camada escondida e camada de saída
    bias_oh = rand(O, 1) - 0.5;    

    % ---------------------- Aplicação do Algoritmo WTA ----------------------    
    C = wta(X_train, C, eta_gaussian);

    % ---------------------- Determinação da abertura dos neurônios escondidos ----------------------

    % Considera os N/2 neurônios mais próximos para cálculo da abertura de
    % cada neurônio
    T = floor(H/2);
    % Vetor que irá armazenar a abertura para cada neurônio da camada escôndida
    sigmas = zeros(H, 1);
    % Percorre todos os neurônios da camada escondida
    distancesBetweenHiddenNeurons = zeros(H, H) + realmax;    
    % Computa a distância entre cada par de neurônios 
    for i=1:H
        for j=i+1:H            
            distanceBetweenNeuronsIandJ = sqrt(sum((C(i, :) - C(j, :)).^2));            
            distancesBetweenHiddenNeurons(i, j) = distanceBetweenNeuronsIandJ;
            distancesBetweenHiddenNeurons(j, i) = distanceBetweenNeuronsIandJ;
        end
    end
    
    % Computa a abertura de cada neurônio escondido     
    % Percorre todos os neurônios da camada escondida
    for i=1:H
        % Vetor que irá armazenar as T menores distâncias do neurônio i em
        % relação aos outros neurônios escondidos
        minDistances = zeros(T, 1);        
        for j=1:T                        
            [minValue, minPosition] = min(distancesBetweenHiddenNeurons(i, :));
            minDistances(j) = minValue;
            distancesBetweenHiddenNeurons(i, minPosition) = realmax;            
        end
        sigmas(i) = sum(minDistances)/T;
    end    
     
    % ---------------------- Treinamento da camada de saída ----------------------    
    while currentEpoch <= maxEpochs    
        trainingPredictions = zeros(O, numberOfTrainingInstances);
        validationPredictions = zeros(O, numberOfValidationInstances);
        for i=1:numberOfTrainingInstances          
             % ------- Hidden Layer -------            
             mi_h = sqrt(sum((X_train(:, i) - C').^2))'; %OK             
             Y_h = exp(-((mi_h.^2)./((2*sigmas).^2)));             
             % ------- Output Layer -------    
             net_o = Woh * Y_h + bias_oh * ones(1, size(Y_h, 2));
             Y_net = exp(net_o)/sum(exp(net_o));  % Aplicação da softmax                          
             E = (-1).*sum((Y_train(:, i).*log(Y_net)));  % Computação do erro                
             trainingPredictions(:, i) = Y_net;
             % backward                 
             df =  (Y_train(:, i)-Y_net);             
             delta_bias_oh = eta * sum((E.*df)')';             
             delta_Woh = eta * (E.*df)*Y_h';       
             
             % update weights  
             Woh = Woh + delta_Woh;
             bias_oh = bias_oh + delta_bias_oh;            
           
        end                       
        error = sum(((Y_train .* (1-trainingPredictions)).^2), 'all')/numberOfTrainingInstances;
        %sprintf("%f", error)
        errors(currentEpoch) = error;
        
        % Validação
        for i=1:numberOfValidationInstances 
              % ------- Hidden Layer -------            
             mi_h = sqrt(sum((X_val(:, i) - C').^2))'; %OK             
             Y_h = exp(-((mi_h.^2)./((2*sigmas).^2)));             
             % ------- Output Layer -------    
             net_o = Woh * Y_h + bias_oh * ones(1, size(Y_h, 2));
             Y_net = exp(net_o)/sum(exp(net_o));  % Aplicação da softmax                                                 
             validationPredictions(:, i) = Y_net;
        end
        validationError = sum(((Y_val .* (1-validationPredictions)).^2), 'all')/numberOfValidationInstances;         
        %sprintf("%f", validationError);
        validationErrors(currentEpoch) = validationError;

        currentEpoch = currentEpoch + 1;
   end     

    finalErrors = errors;
    finalValErrors = validationErrors;
    hiddenVsInputWeights = C;   
    outputVsHiddenWeights = Woh;
    outputVsHiddenBias = bias_oh;
    sigmas = sigmas;
end