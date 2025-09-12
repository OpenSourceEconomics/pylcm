- Why do I need this complex recursive import handling in `user_model`. And can this be done in a different module since the user_model module will be one of the main modules because of the changes?
- I do not like that the post init of user_model is so large now and that it contains a lot of functionality that would not be expected there. Can you create another module (maybe a helper module?) that exports a function which is then called inside the user_mdoules post init fnc?
- Can you write a function that performs a frozendataclass robust setting of data that is then used for all instances where we set an attribute in a frozen dataclass?
- Remove the "demo" module
- I am getting 99 deprecation warnings because in the test suite you use get_lcm_function all the time. Can you change all the tests to use the new approach whereever possible? Maybe keep one (or a few) test that actually check that it still works with the get_lcm_function approach? Ah, I see you already do that in test_model_methods.py -- maybe be a bit more explicit in the module docstring.

