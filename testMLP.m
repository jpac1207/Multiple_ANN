% Realiza predição da classe de um dado padrão de entrada 'X', utilizando
% os pesos aprendidos por cada rede neural 
function [Y, prob] = testMLP(X)
    basic_information = load('.\ann_weights\basic_info.mat');
    neuralNetworksCount = basic_information.neuralNetworksCount;
    outputSize = basic_information.outputSize;
    Y_net_sum = zeros(outputSize, 1);

    for i=1:neuralNetworksCount  
        neuralNetworkParameters = load(['.\ann_weights\ann_weights_' int2str(i) '.mat']);
        % ------- Hidden Layer -------      
        net_h = neuralNetworkParameters.hiddenVsInputWeights * X + neuralNetworkParameters.hiddenVsInputBias * ones(1, size(X, 2));
        Yh = activation(activationType, net_h);
        % ------- Output Layer -------
        net_o = neuralNetworkParameters.outputVsHiddenWeights * Yh + neuralNetworkParameters.outputVsHiddenBias * ones(1, size (Yh, 2));
        Y_net = exp(net_o)./sum(exp(net_o))   % Aplicação da softmax   
        Y_net_sum = Y_net_sum + Y_net;
    end

    [~, index] = max(Y_net_sum);
    Y = index;
    prob = Y_net(index);
end