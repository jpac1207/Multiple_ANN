% Retorna a posição na matriz 'hiddenNeurons' cujo elemento é mais próximo
% do vetor coluna 'inputPattern'
function[minPosition] = getNearestNeuronPosition(inputPattern, hiddenNeurons)
    differences = zeros(size(hiddenNeurons, 1), 1);    
    for j = 1:size(hiddenNeurons, 1)           
        absoluteDifference  = sum((inputPattern - hiddenNeurons(j, :)').^2);
        differences(j) = absoluteDifference;
    end       
    [~, minPosition] = min(differences);     
end