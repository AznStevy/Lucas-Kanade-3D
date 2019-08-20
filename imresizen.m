function Y = imresizen(data,scaling,varargin)
    num_dimensions = ndims(data);
    scaling(1,1:num_dimensions) = scaling(:).';

    data_size = size(data);
    xvec = cell(1,num_dimensions);
    yvec = cell(1,num_dimensions);
    szy = nan(1,num_dimensions);

    nonsing = true(1, num_dimensions);

    for i = 1:num_dimensions
        n = data_size(i);

        if n==1 %for vector input
            nonsing(i)=0;
            szy(i)=1;
            continue;
        end

        szy(i) = round(data_size(i)*scaling(i));  
        m = szy(i);

        xax = linspace(1/n/2, 1-1/n/2 ,n); 
        xax = xax-.5;

        yax = linspace(1/m/2, 1-1/m/2 ,m); 
        yax = yax-.5;

        xvec{i} = xax; 
        yvec{i} = yax;     

    end

     xvec = xvec(nonsing);
     yvec = yvec(nonsing);

     F = griddedInterpolant(xvec,squeeze(data),varargin{:});
     Y = reshape(F(yvec),szy);
end