function I = map_pzy(A)  
  
% ����ά���� A ӳ�䵽 0~255 ��  
  
Min = min(min(A));  
Max = max(max(A));  
if Max == Min  
    I = ones(size(A));  
else  
    I = (A - Min) / (Max - Min);  
end  
I = uint8(I*255);  