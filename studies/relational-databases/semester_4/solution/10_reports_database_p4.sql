DECLARE @JobID BINARY(16);
DECLARE @JobName NVARCHAR(128);
SET @JobName = @JobName;

EXEC msdb.dbo.sp_add_job
    @job_name = N'Trigger Copy Table',
    @enabled = 1,
    @notify_level_eventlog = 0,
    @notify_level_email = 0,
    @notify_level_netsend = 0,
    @notify_level_page = 0,
    @delete_level = 0,
    @description = N'Trigger CreateMirrorTable every 24 h.',
    @category_name = N'[Uncategorized (Local)]',
    @owner_login_name = N'sa',
    @job_id = @JobID OUTPUT;

EXEC msdb.dbo.sp_add_jobstep
    @job_id = @JobID,
    @step_name = N'Trigger Copy Table',
    @subsystem = N'TSQL',
    @command = N'EXEC FoodCourtReports.dbo.CreateMirrorTable 
                  @SourceSchema = ''Orders'', 
                  @TableName = ''Orders''',
    @cmdexec_success_code = 0,
    @on_success_action = 1,
    @on_fail_action = 2,
    @retry_attempts = 0,
    @retry_interval = 0,
    @database_name = N'FoodCourtReports',
    @flags = 0;

DECLARE @ScheduleID INT;
EXEC msdb.dbo.sp_add_jobschedule
    @job_id = @JobID,
    @name = N'Every 24h',
    @enabled = 1,
    @freq_type = 4,
    @freq_interval = 1,
    @freq_subday_type = 1,
    @freq_subday_interval = 0,
    @freq_relative_interval = 0,
    @freq_recurrence_factor = 1,
    @active_start_date = 20231226,
    @active_end_date = 99991231,
    @active_start_time = 010000, 
    @active_end_time = 235959,
    @schedule_id = @ScheduleID OUTPUT;

EXEC msdb.dbo.sp_attach_schedule
   @job_id = @JobID,
   @schedule_id = @ScheduleID;

SELECT @JobID = job_id FROM msdb.dbo.sysjobs WHERE (name = @JobName);
EXEC msdb.dbo.sp_add_jobserver @job_id = @JobID, @server_name = N'(local)';
