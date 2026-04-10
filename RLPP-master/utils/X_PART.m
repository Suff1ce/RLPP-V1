function ensemble = X_PART(x,i,n,h)
%Take corresponding ensemble with given M1 indexes
%   INPUT:
%       x   - x_full, full ensemble
%       i   - required M1 indexes. n * 1 vector
%       n   - total M1 numbers
%   OUTPUT:
%       ensemble  - corresponding ensemble

if isrow(i)
    i = i';
end

ensemble = x( ...
    sort( ...
        reshape( ...
            i+n*(0:h),length(i)*(h+1),1 ...
        ) ...
    ),: ...
);
end
