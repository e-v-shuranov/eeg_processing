from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
event_acc = EventAccumulator('logs')
# /media/hdd/evgeniy/eeg_processing/logs

event_acc.Reload()
# Show all tags in the log file
print(event_acc.Tags())

# E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
# w_times, step_nums, vals = zip(*event_acc.Scalars('Accuracy'))

w_times, step_nums, vals = zip(*event_acc.Scalars('Loss/train'))
# print('w_times: ',w_times)
print('step_nums', step_nums)
print('vals', vals)

w_times, step_nums, vals = zip(*event_acc.Scalars('Loss/test'))
print('vals', vals)