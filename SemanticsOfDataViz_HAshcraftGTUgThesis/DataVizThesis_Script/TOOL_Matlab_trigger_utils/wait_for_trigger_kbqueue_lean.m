%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Robust triggering mechanism
%%%% using KbQueue so we don't miss any short pulse triggers
%%%% 2024, asathe@mit.edu
%%%% https://github.mit.edu/EvLab/TOOL_Matlab_trigger_utils/blob/main/README.md
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function run_start_time = wait_for_trigger_kbqueue_lean.m(delay_milliseconds, esc_key_latency)
  arguments
      % delay polling the queue after creation of the queue
      % by this amount of time, to avoid missing the first event
      % (though this will almost never happen as we start the
      % stim script before we start the scanner, and that operation
      % introduces a natural and sufficient buffer). note that this
      % has no bearing on the script latency relative to onset time
      % (since trigger)
      delay_milliseconds = 100,
      % we will use a blocking trigger check call to KbEventGet
      % for these many seconds, and only check for ESC key being
      % pressed in between. so this is how long you'll have to
      % press down the ESC key for, in seconds, to ESC out of 
      % the script. making this too low will introduce latency
      % in the script onset. making this high will make ESCing
      % very difficult at trigger time. 
      % note that this again has no bearing on detection of the
      % actual trigger since keyboard events are buffered in a
      % queue and always processed
      esc_key_latency = 1
  end
  
  KbName('UnifyKeyNames')

  fprintf('--------- CALLED `wait_for_trigger_kbqueue`\n')

  % 'keyList' is an optional 256-length vector of doubles (not logicals)  
  %  with each element corresponding to a particular key (use [KbName](KbName)  
  %  to map between keys and their positions). If the double value  
  %  corresponding to a particular key is zero, events for that key  
  %  are not added to the queue and will not be reported.
  % see also: https://github.com/caomw/Psychtoolbox-3/blob/master/Psychtoolbox/PsychBasic/KbTriggerWait.m
  keyList = zeros(1, 256, 'double');
  % keyList(KbName('=+')) = 1.0;
  keyList(KbName('t')) = 1.0;
    
  dvc = find_device;

  KbQueueCreate(dvc, keyList);
  KbQueueStart(dvc);
  KbQueueFlush(dvc);
  WaitSecs(double(delay_milliseconds) / 1000.0);
  
  fprintf('WAITING FOR TRIGGER on device = %d.\nTo ESC, first trigger and then Ctrl+C and "sca"\n', dvc);

  run_start_time = KbQueueWait(dvc);
  KbQueueRelease(dvc);
  
  log_time;
  RestrictKeysForKbCheck([]);
      
end
