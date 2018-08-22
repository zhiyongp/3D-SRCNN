function I = map_pzy(A)  
  
% 将二维数组 A 映射到 0~255 中  
  
Min = min(min(A));  
Max = max(max(A));  
if Max == Min  
    I = ones(size(A));  
else  
    I = (A - Min) / (Max - Min);  
end  
I = uint8(I*255);  