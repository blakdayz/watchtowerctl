Feature: Incident Response

  Scenario: Detect and respond to a security incident
    Given the user has detected an unusual process
    When they run the command "sudo dtruss -r <pid>"
    Then the system calls for that process are tracked and logged
