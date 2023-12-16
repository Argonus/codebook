USE FoodCourt
GO

DECLARE @Today DATE = GETDATE();
DECLARE @FirstDayOfNextMonth DATE = DATEADD(MONTH, DATEDIFF(MONTH, 0, @Today) + 1, 0);
DECLARE @LastDayOfNextMonth DATE = DATEADD(DAY, -1, DATEADD(MONTH, DATEDIFF(MONTH, 0, @FirstDayOfNextMonth) + 1, 0));

WITH DateRange AS (
	SELECT @FirstDayOfNextMonth AS DayDate
	UNION ALL
	SELECT DATEADD(DAY, 1, DayDate) FROM DateRange WHERE DayDate < @LastDayOfNextMonth
), EmployeeSchedules AS (
    SELECT * FROM Staff.Schedules WHERE EmployeeId = 361 AND LocationID = 128
), RecursiveSchedules AS (
    SELECT
	    ES.ID
        , ES.DayOfWeek
        , ES.RepeatFrequency
        , ES.StartDate
		, ES.EndDate
    FROM EmployeeSchedules AS ES
    WHERE 
		ES.StartDate <= @FirstDayOfNextMonth 
		AND ES.EndDate >= @LastDayOfNextMonth

    UNION ALL

    SELECT
		RS.ID
        , RS.DayOfWeek
        , RS.RepeatFrequency
        , DATEADD(WEEK, RS.RepeatFrequency, RS.StartDate) AS StartDate
		, RS.EndDate
    FROM RecursiveSchedules AS RS
    WHERE DATEADD(WEEK, RS.RepeatFrequency, RS.StartDate) <= RS.EndDate
), MidStepWorkingHours AS (
	SELECT
		RS.DayOfWeek
		, RS.StartDate AS [Date]
		, CONVERT(TIME(0), SWH.StartTime) AS StartTime
		, CONVERT(TIME(0), SWH.EndTime) AS EndTime
	FROM RecursiveSchedules AS RS
	INNER JOIN DateRange AS DR
		ON RS.StartDate = DR.DayDate
	INNER JOIN Staff.WorkingHours AS SWH
		ON SWH.ScheduleId = RS.ID
)
SELECT
    DATENAME(WEEKDAY, MSWH.Date) AS DayOfWeek
    , MSWH.Date AS [DATE]
    , STRING_AGG(CONCAT(StartTime, '-', EndTime), ', ') AS WorkingHours
FROM MidStepWorkingHours AS MSWH
GROUP BY DATENAME(WEEKDAY, MSWH.Date), MSWH.Date
ORDER BY [DATE];