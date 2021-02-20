function [PositionWristAligned] = Alignement(PositionWrist,INDEX)
%This function Align each trial on the origin coordinate. In the new
%reference frame, each trial start from 0,0,0. it gives as Output the
%Aligned vector gien in input
for i = 1:size(PositionWrist,3)
    if ~isnan(PositionWrist(1,:,i) )
          nonannnn(i) = [i];
    PositionWristAligned(:,:,i) = PositionWrist(:,:,i) - PositionWrist(INDEX,:,i);
    
    else
       % nannnn(i) = [i];
        gggg = ~isnan(PositionWrist(:,:,i));
        ggggg = find(gggg ~= 0 & gggg ~= NaN);
        if ~isempty(ggggg)
        PositionWristAligned(:,:,i) = PositionWrist(:,:,i) - PositionWrist(ggggg(1,1),:,i);
        PositionWristAligned(1:ggggg(1,1),:,i) = 0;
        else
            PositionWristAligned(:,:,i) = NaN;
    end
    gggg = []; ggggg = [];
end
end

