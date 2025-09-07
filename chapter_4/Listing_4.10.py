booking_flight_example = dspy.Example(
    message="I want to book a flight", 
    labels=unique_intents, 
    intent_label='Flight Booking Request').with_inputs("message", "labels")
booking_flight_prediction = Prediction(intent_label='Flight Booking Request')
print(validate_answer(booking_flight_example, booking_flight_prediction))
