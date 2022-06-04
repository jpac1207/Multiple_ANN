% Realiza o treinamento da MLP, de acordo com os parametros:
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
function [hiddenVsInputWeights, hiddenVsInputBias, outputVsHiddenWeights, outputVsHiddenBias, finalErrors, finalValErrors, trainingFinalPredictions, validationFinalPredictions] = trainMLP(I, H, O, maxEpochs, eta, ...
    activationType, X_train, Y_train, X_val, Y_val, flag)
    currentEpoch = 1;    
    numberOfTrainingInstances = size(X_train, 2);
    numberOfValidationInstances = size(X_val, 2); 
    errors = zeros(maxEpochs, 1);  
    validationErrors = zeros(maxEpochs, 1);  
    % Init weights    
    Whi = rand(H, I) - 0.5;
    bias_hi = rand(H, 1) - 0.5;   
    if(flag)        
         Woh = zeros(O, H);
         bias_oh = zeros(O, 1); 
    else
         Woh = rand (O, H) - 0.5;
         bias_oh = rand(O, 1) - 0.5; 
    end
    %Woh = rand (O, H) - 0.5;
    %bias_oh = rand(O, 1) - 0.5;    
    
    while currentEpoch <= maxEpochs       
        trainingPredictions = zeros(O, numberOfTrainingInstances);
        validationPredictions = zeros(O, numberOfValidationInstances);
        for i=1:numberOfTrainingInstances
            % ------- Hidden Layer -------      
            net_h = Whi * X_train(:, i) + bias_hi * ones(1, size(X_train(:, i), 2));
            Yh = activation(activationType, net_h);
            % ------- Output Layer -------
            net_o = Woh * Yh + bias_oh * ones(1, size (Yh, 2));
            Y_net = exp(net_o)./sum(exp(net_o));   % Aplicação da softmax      
            trainingPredictions(:, i) = Y_net;
            E = ((-1).*sum((Y_train(:, i).*log(Y_net))));  % Computação do erro                   
            %sprintf("%f", E);             
            
            % backward    
            df =  (Y_train(:, i)-Y_net);
            %df
            delta_bias_oh = eta * sum((E.*df)')';
            delta_Woh = eta * (E.*df)*Yh';
            Eh = (Woh')*(E.*df);
            
            df = activationDerivative(activationType, net_h);
            delta_bias_hi = (eta) * sum((Eh.*df)')';
            delta_Whi = (eta) * (Eh.*df) * X_train(:, i)';
        
            %update weights  
            Whi = Whi + delta_Whi;   
            bias_hi = bias_hi + delta_bias_hi;   
            Woh = Woh + delta_Woh;
            bias_oh = bias_oh + delta_bias_oh;      
        end       
        %calculate error                          
        error = sum(((Y_train .* (1-trainingPredictions)).^2), 'all')/numberOfTrainingInstances;
        errors(currentEpoch) = error;
        
         % Validação
        for i=1:numberOfValidationInstances
            % ------- Validation -------
            val_net_h = Whi * X_val(:, i) + bias_hi * ones(1, size(X_val(:, i), 2));
            val_Yh = activation(activationType, val_net_h);    
            val_net_o = Woh * val_Yh + bias_oh * ones(1, size (val_Yh, 2));        
            val_Y_net = exp(val_net_o)./sum(exp(val_net_o));   
            validationPredictions(:, i) = val_Y_net;
            %---------------------------
        end
        validationError = sum(((Y_val .* (1-validationPredictions)).^2), 'all')/numberOfValidationInstances;               
        validationErrors(currentEpoch) = validationError;
        currentEpoch = currentEpoch + 1;
   end     
    
    finalErrors = errors;
    finalValErrors = validationErrors;
    trainingFinalPredictions = trainingPredictions;
    validationFinalPredictions = validationPredictions;
    hiddenVsInputWeights = Whi;
    hiddenVsInputBias = bias_hi;
    outputVsHiddenWeights = Woh;
    outputVsHiddenBias = bias_oh;
end
% Aplica uma função de ativação no parâmetro 'value' utilizando a flag 'type'
function f = activation(type, value)
    if(type == 0)
        f = logsig(value);
    else
        f = tanh(value);
    end
end

% Computa a derivada de 'value' utilizando a função definida pela flag 'type'
function f = activationDerivative(type, value)
    if(type == 0)
        f = logsig(value) - (logsig(value).^2);
    else
        f = 1 - (tanh(value).^2);
    end
end