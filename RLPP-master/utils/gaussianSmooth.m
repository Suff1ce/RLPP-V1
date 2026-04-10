function smoothedSignal = gaussianSmooth(signal,kernelSize)
%Smooth the signal with gaussian kernel
%   INPUT:
%       signal          - signal to be smoothed
%       kernelSize      - gaussian kernel size. Length of 1 sigma
%   OUTPUT:
%       smoothedSignal  - smoothed signal

smoothedSignal = smoothdata(signal, 'gaussian', 5*kernelSize+1);
end
