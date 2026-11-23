

run_start_time = wait_for_trigger_kbqueue;
now = GetSecs;
fprintf('\nLatency: %.4f milliseconds\n', 1000 * (now - run_start_time));