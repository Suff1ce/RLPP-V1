function [lowRaster,highRaster] = getRaster(allActions,spikes)
flag = 0;
idx = 1;
lowRaster = zeros(size(spikes,1),1,101);
for i=2:length(allActions)
    if (allActions(i-1)==0 && allActions(i)==1)
        start = i;
    end
    if (allActions(i-1)==2 && allActions(i)==0)
        stop = i-1;
        flag = 1;
    end
    if (flag==1)
        temp = double(spikes(:,start:stop));
        temp = [temp(:,allActions(start:stop)>0) spikes(:,stop+(1:20))];
        for j = 1:size(spikes,1)
            lowRaster(j,idx,:) = temp(j,1:101);
        end
        idx = idx+1;
        flag = 0;
    end
end

flag = 0;
idx = 1;
highRaster = zeros(size(spikes,1),1,101);
for i=2:length(allActions)
    if (allActions(i-1)==0 && allActions(i)==1)
        start = i;
    end
    if (allActions(i-1)==3 && allActions(i)==0)
        stop = i-1;
        flag = 1;
    end
    if (flag==1)
        temp = double(spikes(:,start:stop));
        temp = [temp(:,allActions(start:stop)>0) spikes(:,stop+(1:20))];
        for j = 1:size(spikes,1)
            highRaster(j,idx,:) = temp(j,1:101);
        end
        idx = idx+1;
        flag = 0;
    end
end
end
