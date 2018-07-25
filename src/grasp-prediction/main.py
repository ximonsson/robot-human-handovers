def __main__ ():
    data = load_data ()
    model = train (data)
    validate (model, data)
    accuracy = test (model, data)
