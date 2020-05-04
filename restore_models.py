import tensorflow as tf
import os
import numpy as np

def restore(session, save_file, raise_if_not_found=False, copy_mismatched_shapes=False):
    if not os.path.exists(save_file) and raise_if_not_found:
        raise Exception('File %s not found' % save_file)
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])
    var_name_to_var = {var.name : var for var in tf.global_variables()}
    restore_vars = []
    restored_var_names = set()
    restored_var_new_shape = []
    print('Restoring:')
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        for var_name, saved_var_name in var_names:
            if 'global_step' in var_name:
                restored_var_names.add(saved_var_name)
                continue
            curr_var = var_name_to_var[var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
                # print(str(saved_var_name) + ' -> \t' + str(var_shape) + ' = ' +
                #       str(int(np.prod(var_shape) * 4 / 10**6)) + 'MB')
                restored_var_names.add(saved_var_name)
            else:
                print('Shape mismatch for var', saved_var_name, 'expected', var_shape,
                      'got', saved_shapes[saved_var_name])
                restored_var_new_shape.append((saved_var_name, curr_var, reader.get_tensor(saved_var_name)))
                print('bad things')
    ignored_var_names = sorted(list(set(saved_shapes.keys()) - restored_var_names))
    print('\n')
    if len(ignored_var_names) == 0:
        print('Restored all variables')
    else:
        pass
        # print('Did not restore:' + '\n\t'.join(ignored_var_names))

    if len(restore_vars) > 0:
        saver = tf.train.Saver(restore_vars)
        saver.restore(session, save_file)

    if len(restored_var_new_shape) > 0 and copy_mismatched_shapes:
        print('trying to restore misshapen variables')
        assign_ops = []
        for name, kk, vv in restored_var_new_shape:
            copy_sizes = np.minimum(kk.get_shape().as_list(), vv.shape)
            slices = [slice(0,cs) for cs in copy_sizes]
            print('copy shape', name, kk.get_shape().as_list(), '->', copy_sizes.tolist())
            new_arr = session.run(kk)
            new_arr[slices] = vv[slices]
            assign_ops.append(tf.assign(kk, new_arr))
        session.run(assign_ops)
        print('Copying unmatched weights done')
    print('Restored %s' % save_file)
    try:
        start_iter = int(save_file.split('-')[-1])
    except ValueError:
        print('Could not parse start iter, assuming 0')
        start_iter = 0
    return start_iter


def restore_from_dir(sess, folder_path, raise_if_not_found=False, copy_mismatched_shapes=False):
    start_iter = 0
    ckpt = tf.train.get_checkpoint_state(folder_path)
    if ckpt and ckpt.model_checkpoint_path:
        print('Restoring')
        start_iter = restore(sess, ckpt.model_checkpoint_path, raise_if_not_found, copy_mismatched_shapes)
    else:
        if raise_if_not_found:
            raise Exception('No checkpoint to restore in %s' % folder_path)
        else:
            print('No checkpoint to restore in %s' % folder_path)
    return start_iter