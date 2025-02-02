from behave import given, when, then


@given("the user has detected an unusual process")
def step_impl(context):
    # Implement the logic to detect an unusual process
    pass


@when('they run the command "sudo dtruss -r <pid>"')
def step_impl(context):
    # Run the command using subprocess or similar
    context.cmd_result = subprocess.run(["sudo", "dtruss", "-r", "<pid>"])


@then("the system calls for that process are tracked and logged")
def step_impl(context):
    # Check if the command output is as expected
    assert "Expected log output" in context.cmd_result.stdout.decode()
