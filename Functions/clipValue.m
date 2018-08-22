function val = clipValue(val, valMin, valMax)
% check if value is valid
val(val>valMax) = valMax;
val(val<valMin) = valMin;

end