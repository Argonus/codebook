USE [master]
GO
/****** Object:  Database [FoodCourtReports]    Script Date: 04.01.2024 20:43:22 ******/
CREATE DATABASE [FoodCourtReports]
 CONTAINMENT = NONE
 ON  PRIMARY 
( NAME = N'FoodCourtReports', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL15.MSSQLSERVER\MSSQL\DATA\FoodCourtReports.mdf' , SIZE = 8192KB , MAXSIZE = UNLIMITED, FILEGROWTH = 65536KB )
 LOG ON 
( NAME = N'FoodCourtReports_log', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL15.MSSQLSERVER\MSSQL\DATA\FoodCourtReports_log.ldf' , SIZE = 8192KB , MAXSIZE = 2048GB , FILEGROWTH = 65536KB )
 WITH CATALOG_COLLATION = DATABASE_DEFAULT
GO
ALTER DATABASE [FoodCourtReports] SET COMPATIBILITY_LEVEL = 150
GO
IF (1 = FULLTEXTSERVICEPROPERTY('IsFullTextInstalled'))
begin
EXEC [FoodCourtReports].[dbo].[sp_fulltext_database] @action = 'enable'
end
GO
ALTER DATABASE [FoodCourtReports] SET ANSI_NULL_DEFAULT OFF 
GO
ALTER DATABASE [FoodCourtReports] SET ANSI_NULLS OFF 
GO
ALTER DATABASE [FoodCourtReports] SET ANSI_PADDING OFF 
GO
ALTER DATABASE [FoodCourtReports] SET ANSI_WARNINGS OFF 
GO
ALTER DATABASE [FoodCourtReports] SET ARITHABORT OFF 
GO
ALTER DATABASE [FoodCourtReports] SET AUTO_CLOSE OFF 
GO
ALTER DATABASE [FoodCourtReports] SET AUTO_SHRINK OFF 
GO
ALTER DATABASE [FoodCourtReports] SET AUTO_UPDATE_STATISTICS ON 
GO
ALTER DATABASE [FoodCourtReports] SET CURSOR_CLOSE_ON_COMMIT OFF 
GO
ALTER DATABASE [FoodCourtReports] SET CURSOR_DEFAULT  GLOBAL 
GO
ALTER DATABASE [FoodCourtReports] SET CONCAT_NULL_YIELDS_NULL OFF 
GO
ALTER DATABASE [FoodCourtReports] SET NUMERIC_ROUNDABORT OFF 
GO
ALTER DATABASE [FoodCourtReports] SET QUOTED_IDENTIFIER OFF 
GO
ALTER DATABASE [FoodCourtReports] SET RECURSIVE_TRIGGERS OFF 
GO
ALTER DATABASE [FoodCourtReports] SET  ENABLE_BROKER 
GO
ALTER DATABASE [FoodCourtReports] SET AUTO_UPDATE_STATISTICS_ASYNC OFF 
GO
ALTER DATABASE [FoodCourtReports] SET DATE_CORRELATION_OPTIMIZATION OFF 
GO
ALTER DATABASE [FoodCourtReports] SET TRUSTWORTHY OFF 
GO
ALTER DATABASE [FoodCourtReports] SET ALLOW_SNAPSHOT_ISOLATION OFF 
GO
ALTER DATABASE [FoodCourtReports] SET PARAMETERIZATION SIMPLE 
GO
ALTER DATABASE [FoodCourtReports] SET READ_COMMITTED_SNAPSHOT OFF 
GO
ALTER DATABASE [FoodCourtReports] SET HONOR_BROKER_PRIORITY OFF 
GO
ALTER DATABASE [FoodCourtReports] SET RECOVERY FULL 
GO
ALTER DATABASE [FoodCourtReports] SET  MULTI_USER 
GO
ALTER DATABASE [FoodCourtReports] SET PAGE_VERIFY CHECKSUM  
GO
ALTER DATABASE [FoodCourtReports] SET DB_CHAINING OFF 
GO
ALTER DATABASE [FoodCourtReports] SET FILESTREAM( NON_TRANSACTED_ACCESS = OFF ) 
GO
ALTER DATABASE [FoodCourtReports] SET TARGET_RECOVERY_TIME = 60 SECONDS 
GO
ALTER DATABASE [FoodCourtReports] SET DELAYED_DURABILITY = DISABLED 
GO
ALTER DATABASE [FoodCourtReports] SET ACCELERATED_DATABASE_RECOVERY = OFF  
GO
EXEC sys.sp_db_vardecimal_storage_format N'FoodCourtReports', N'ON'
GO
ALTER DATABASE [FoodCourtReports] SET QUERY_STORE = OFF
GO
USE [FoodCourtReports]
GO
/****** Object:  Schema [INGESTION]    Script Date: 04.01.2024 20:43:22 ******/
CREATE SCHEMA [INGESTION]
GO
/****** Object:  Table [INGESTION].[Orders]    Script Date: 04.01.2024 20:43:22 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [INGESTION].[Orders](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[LocationID] [smallint] NOT NULL,
	[Status] [varchar](25) NOT NULL,
	[DeliveryAddressID] [int] NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
	[DeletedAt] [datetime2](7) NULL,
	[OrderType] [varchar](25) NOT NULL,
	[RequestedDateTime] [datetime] NULL,
	[CustomerID] [int] NULL
) ON [PRIMARY]
GO
/****** Object:  StoredProcedure [dbo].[CreateMirrorTable]    Script Date: 04.01.2024 20:43:22 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE PROCEDURE [dbo].[CreateMirrorTable]
    @SourceSchema NVARCHAR(128),
    @TableName NVARCHAR(128)
AS
BEGIN
    SET NOCOUNT ON;

    DECLARE @CHECK_QUERY NVARCHAR(MAX);
    DECLARE @CREATE_QUERY NVARCHAR(MAX);
	
	DECLARE @TargetSchema NVARCHAR(256);
	SET @TargetSchema = N'INGESTION';

    SET @CHECK_QUERY = N'SELECT @TableExists = 
							CASE 
								WHEN EXISTS (SELECT * FROM sys.tables WHERE name = @TableName AND SCHEMA_NAME(schema_id) = @TargetSchema) THEN 1 
								ELSE 0 
							END';

	DECLARE @TableExists BIT;
    EXEC sp_executesql @CHECK_QUERY, N'@TableName NVARCHAR(128), @TargetSchema NVARCHAR(256), @TableExists BIT OUTPUT', @TableName, @TargetSchema, @TableExists OUTPUT;

    IF @TableExists = 0
    BEGIN
        SET @CREATE_QUERY = N'SELECT * INTO FoodCourtReports.' + QUOTENAME(@TargetSchema) + '.' + QUOTENAME(@TableName) + 
                            N' FROM FoodCourt.' + QUOTENAME(@SourceSchema) + '.' + QUOTENAME(@TableName) + ' WHERE 1 = 0;';
		PRINT(@CREATE_QUERY);		
        EXEC sp_executesql @CREATE_QUERY;
    END
	ELSE
	BEGIN
		PRINT('Table ' + @TableName + ' already exsits');
	END
END;
GO
/****** Object:  StoredProcedure [dbo].[UpsertData]    Script Date: 04.01.2024 20:43:22 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE   PROCEDURE [dbo].[UpsertData]
    @SourceSchema NVARCHAR(128),
    @SourceTableName NVARCHAR(128),
    @TargetSchema NVARCHAR(128),
    @TargetTableName NVARCHAR(128),
    @PrimaryKey NVARCHAR(128)
AS
BEGIN
    SET NOCOUNT ON;

    DECLARE @QUERY NVARCHAR(MAX);
    DECLARE @ColumnList NVARCHAR(MAX) = '';
    DECLARE @ColumnUpdateList NVARCHAR(MAX) = '';

    SELECT @ColumnList = STRING_AGG(QUOTENAME(column_name), ', ')
    FROM FoodCourt.INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = @SourceSchema AND TABLE_NAME = @SourceTableName
    AND column_name <> @PrimaryKey;

    SELECT @ColumnUpdateList = STRING_AGG('Target.' + QUOTENAME(column_name) + ' = Source.' + QUOTENAME(column_name), ', ')
    FROM FoodCourt.INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = @SourceSchema AND TABLE_NAME = @SourceTableName
    AND column_name <> @PrimaryKey;

    SET @QUERY = 'SET IDENTITY_INSERT FoodCourtReports.' + QUOTENAME(@TargetSchema) + '.' + QUOTENAME(@TargetTableName) + ' ON;' +
				 'MERGE INTO FoodCourtReports.' + QUOTENAME(@TargetSchema) + '.' + QUOTENAME(@TargetTableName) + ' AS Target ' +
                 'USING (SELECT * FROM FoodCourt.' + QUOTENAME(@SourceSchema) + '.' + QUOTENAME(@SourceTableName) + ') AS Source ' +
                 'ON Target.' + QUOTENAME(@PrimaryKey) + ' = Source.' + QUOTENAME(@PrimaryKey) + ' ' +
                 'WHEN MATCHED THEN UPDATE SET ' + @ColumnUpdateList + ' ' +
                 'WHEN NOT MATCHED BY TARGET THEN INSERT (' + @ColumnList + ', ' + QUOTENAME(@PrimaryKey) + ') VALUES (Source.' + @ColumnList + ', Source.' + QUOTENAME(@PrimaryKey) + ');';

	PRINT('List one')
	PRINT(@ColumnUpdateList)
	PRINT('List Two')
	PRINT(@ColumnList)
	PRINT(@QUERY)

    EXEC sp_executesql @QUERY;
END;
GO
USE [master]
GO
ALTER DATABASE [FoodCourtReports] SET  READ_WRITE 
GO
