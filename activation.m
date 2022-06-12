% Aplica uma função de ativação no parâmetro 'value' utilizando a flag 'type'
function f = activation(type, value)
    if(type == 0)
        f = logsig(value);
    else
        f = tanh(value);
    end
end