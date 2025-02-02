Feature: Backup and Recovery

  Scenario: Create a secure backup using Time Machine
    Given the user wants to back up their system
    When they run the command "tmutil startbackup --destination /Volumes/ExternalDrive"
    Then the backup starts successfully and is stored on the specified drive
