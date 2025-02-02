import subprocess

from behave import given, when, then, runner


@given("the user wants to back up their system")
def step_impl(context: runner.Context):
    # Set up any necessary context if required
    pass


@when('they run the command "tmutil startbackup --destination /Volumes/ExternalDrive"')
def step_impl(context: runner.Context):
    # Run the command using subprocess or similar
    context.cmd_result = subprocess.run(
        ["tmutil", "startbackup", "--destination", "/Volumes/ExternalDrive"]
    )


@then("the backup starts successfully and is stored on the specified drive")
def step_impl(context: runner.Context):
    # Check if the command was successful (could involve checking system state)
    assert context.cmd_result.returncode == 0
