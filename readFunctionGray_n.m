function I = readFunctionGray_n(filename)

% Read the image from file, and convert it to the gray scale representation
% if it's coloured (i.e. it has color dimension)
I = imread(filename);

[~ , ~, d] = size(I);
if d > 1   
    I = rgb2gray(I);
    
end


