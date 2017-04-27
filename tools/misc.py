
def get_options(args):
    options = dict()
    for section in args:
        for opt_name in args[section]:
            opt_val = args[section][opt_name]
            options[opt_name] = int(opt_val) if opt_val.isdigit() else opt_val

    return options