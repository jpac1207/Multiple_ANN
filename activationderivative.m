% Computa a derivada de 'value' utilizando a função definida pela flag 'type'
function f = activationDerivative(type, value)
    if(type == 0)
        f = logsig(value) - (logsig(value).^2);
    else
        f = 1 - (tanh(value).^2);
    end
end