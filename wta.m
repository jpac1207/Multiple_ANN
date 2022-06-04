% Aplicação do algoritmo WTA. Recebe como argumento, a matrix 'inputMatrix'
% com os padrões de entrada, a matrix 'hiddenNeurons' com os pesos dos
% neurônios escondidos e o eta relativo a atualização dos centros dos neurônios escondidos. Como retorno, são devolvidos:
% 'nearestHiddenNeurons' -> Vetor coluna contendo a posição do neurônio
% mais próximo para cada padrão de entrada;
% 'hiddenNeurons' -> Neurônios escondidos com os valores de centro
% atualizados
function [hiddenNeurons] = wta(inputMatrix, hiddenNeurons, eta_gaussian)    
    previousQuantizationError = realmax;
    howManyIterations = 0;
    maxOfIterations = 100;
    numberOfInstances = size(inputMatrix, 2);    
    while true       
        quantizationError = 0;
        % Percorre todos os vetores de entrada x
        for i = 1:numberOfInstances                 
           % Para cada vetor de entrada, determina o centro mais próximo                      
           minPosition = getNearestNeuronPosition(inputMatrix(:, i), hiddenNeurons);          
           % Atualiza o centro mais próximo             
           hiddenNeurons(minPosition, :) = hiddenNeurons(minPosition, :) + (eta_gaussian * (inputMatrix(:, i)' - hiddenNeurons(minPosition, :)));                 
           % Computa erro de quantização
           quantizationError = quantizationError + sum(sqrt(((inputMatrix(:, i)' - hiddenNeurons(minPosition, :)).^2)).^2);
        end       
        quantizationError = quantizationError/numberOfInstances;
        howManyIterations = howManyIterations + 1;        
        % Condições de Parada: maxOfIterations ou erro não diminuiu desde a
        % última iteração          
        %howManyIterations
        %sprintf("%f", quantizationError)
        %sprintf("%f", previousQuantizationError)
        if(quantizationError < previousQuantizationError && howManyIterations <= maxOfIterations)
            previousQuantizationError = quantizationError;
        else            
            break;
        end    
    end  
end