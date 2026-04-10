function [rest, press_low, press_high, press_release] = find_behavior_index(EVT01,EVT02,EVT03,EVT04,EVT05,EVT06,EVT07,EVT08)
%Find the behavior time index from the original events from Plexon
% CAUTION BEFORE USE. THE EVENT DEFINE MAY CHANGE
%   INPUT:
%       EVT0x   - event data, EVTnum of rows of [EVTtime 1]
%   OUTPUT:     All the elements are event time indexes
%       rest        - 500ms before start cue to start cue. successTrialNum * 2 matrix
%       press_low   - press low to release. sucessLowNum * 2 matrix
%       press_high  - press high to release. sucessLowNum * 2 matrix
%       press_release   - press to release. N * 2 matrix

%{
EVT signal			Event
    1           HIGH
    2           LOW
	3           Lever held (success)
	4           Lever not held (early release)
	5           Start trial
	6           Press lever			
	7           Release lever
	8           Fail
1/2 + 3~8 together decide one behavior event
%}

%% Remove possible repeated record
EVT01=unique(EVT01(:,1));
EVT02=unique(EVT02(:,1));
EVT03=unique(EVT03(:,1));
EVT04=unique(EVT04(:,1));
EVT05=unique(EVT05(:,1));
EVT06=unique(EVT06(:,1));
EVT07=unique(EVT07(:,1));
EVT08=unique(EVT08(:,1));

%% Detect press events
INDEX_press=[];
% all high press
for i=1:length(EVT01)
  a= find((EVT01(i)>(EVT06-0.001)).*(EVT01(i)<(EVT06+0.001))==1);
  INDEX_press=[INDEX_press a];
end
% all low press
for i=1:length(EVT02)
  a= find((EVT02(i)>(EVT06-0.001)).*(EVT02(i)<(EVT06+0.001))==1);
  INDEX_press=[INDEX_press a];
end
press=EVT06(INDEX_press);

%% Detect release events based on press
release=[];
for i=1:length(press) % two type of release: early release(04) & release(07)
  a = find(EVT04>press(i),1,'first');
  b = find(EVT07>press(i),1,'first');
  % the closer release event corresponds to current press
  if (isempty(a) && ~isempty(b))
    release=[release EVT07(b)];
  elseif (isempty(b) && ~isempty(a))
    release=[release EVT04(a)];
  elseif (~isempty(a) && ~isempty(b))
    release=[release min(EVT04(a),EVT07(b))];
  else
    warning("Unfind release event!");
  end
end
% put press-release together for return
press_release = [press release'];

%% delete early release
ind_early=[];
for i=1:length(EVT04)
  a= find(EVT06<EVT04(i), 1, 'last' );
  ind_early=[ind_early a];
end
EVT06(ind_early)=[];

%% detect high success events
INDEX=[];
for i=1:length(EVT01)
     a= find((EVT01(i)>EVT03-0.001).*(EVT01(i)<=(EVT03+0.001))==1, 1, 'last');
     INDEX=[INDEX a];
end

high_success=EVT03(INDEX);

%% detect low success events
INDEX2=[];
for i=1:length(EVT02)
     a= find((EVT02(i)>EVT03-0.001).*(EVT02(i)<=(EVT03+0.001))==1, 1, 'last');
     INDEX2=[INDEX2 a];
end

low_success=EVT03(INDEX2);

%% detect success high press
INDEX3=[];
for i=1:length(high_success)
     a= find(EVT06<high_success(i), 1, 'last' );
     INDEX3=[INDEX3 a];
end

success_high_press=EVT06(INDEX3);

%% detect success low press
INDEX4=[];
for i=1:length(low_success)
     a= find(EVT06<low_success(i), 1, 'last' );
     INDEX4=[INDEX4 a];
end

success_low_press=EVT06(INDEX4);

%% detect success high start
INDEX5=[];
for i=1:length(high_success)
    a= find(EVT05<high_success(i), 1, 'last' );
    INDEX5=[INDEX5 a];
end

success_high_start=EVT05(INDEX5);

%% detect success low start
INDEX6=[];
for i=1:length(low_success)
    a= find(EVT05<low_success(i), 1, 'last' );
    INDEX6=[INDEX6 a];
end

success_low_start=EVT05(INDEX6);

%% detect success high release
INDEX7=[];
for i=1:length(high_success)
    a= find(EVT07>high_success(i), 1, 'first' );
    INDEX7=[INDEX7 a];
end
success_high_release=EVT07(INDEX7);

%% detect success low release
INDEX8=[];
for i=1:length(low_success)
    a= find(EVT07>low_success(i), 1, 'first' );
    INDEX8=[INDEX8 a];
end
success_low_release=EVT07(INDEX8);

%% Put them together for return
success_start = sort([success_high_start; success_low_start]);
rest = [success_start-0.5 success_start];
press_high = [success_high_press success_high_release];
press_low  = [success_low_press success_low_release];

end