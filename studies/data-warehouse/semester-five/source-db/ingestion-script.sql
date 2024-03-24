USE [master]
GO
/****** Object:  Database [Ingestion]    Script Date: 12.01.2024 16:44:07 ******/
CREATE DATABASE [Ingestion]
 CONTAINMENT = NONE
 ON  PRIMARY 
( NAME = N'FoodCourt', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL15.MSSQLSERVER\MSSQL\DATA\Ingestion.mdf' , SIZE = 15360KB , MAXSIZE = UNLIMITED, FILEGROWTH = 1024KB )
 LOG ON 
( NAME = N'FoodCourt_log', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL15.MSSQLSERVER\MSSQL\DATA\Ingestion.ldf' , SIZE = 31744KB , MAXSIZE = 2097152KB , FILEGROWTH = 1024KB )
 WITH CATALOG_COLLATION = DATABASE_DEFAULT
GO
ALTER DATABASE [Ingestion] SET COMPATIBILITY_LEVEL = 150
GO
IF (1 = FULLTEXTSERVICEPROPERTY('IsFullTextInstalled'))
begin
EXEC [Ingestion].[dbo].[sp_fulltext_database] @action = 'enable'
end
GO
ALTER DATABASE [Ingestion] SET ANSI_NULL_DEFAULT OFF 
GO
ALTER DATABASE [Ingestion] SET ANSI_NULLS OFF 
GO
ALTER DATABASE [Ingestion] SET ANSI_PADDING OFF 
GO
ALTER DATABASE [Ingestion] SET ANSI_WARNINGS OFF 
GO
ALTER DATABASE [Ingestion] SET ARITHABORT OFF 
GO
ALTER DATABASE [Ingestion] SET AUTO_CLOSE OFF 
GO
ALTER DATABASE [Ingestion] SET AUTO_SHRINK OFF 
GO
ALTER DATABASE [Ingestion] SET AUTO_UPDATE_STATISTICS ON 
GO
ALTER DATABASE [Ingestion] SET CURSOR_CLOSE_ON_COMMIT OFF 
GO
ALTER DATABASE [Ingestion] SET CURSOR_DEFAULT  GLOBAL 
GO
ALTER DATABASE [Ingestion] SET CONCAT_NULL_YIELDS_NULL OFF 
GO
ALTER DATABASE [Ingestion] SET NUMERIC_ROUNDABORT OFF 
GO
ALTER DATABASE [Ingestion] SET QUOTED_IDENTIFIER OFF 
GO
ALTER DATABASE [Ingestion] SET RECURSIVE_TRIGGERS OFF 
GO
ALTER DATABASE [Ingestion] SET  DISABLE_BROKER 
GO
ALTER DATABASE [Ingestion] SET AUTO_UPDATE_STATISTICS_ASYNC OFF 
GO
ALTER DATABASE [Ingestion] SET DATE_CORRELATION_OPTIMIZATION OFF 
GO
ALTER DATABASE [Ingestion] SET TRUSTWORTHY OFF 
GO
ALTER DATABASE [Ingestion] SET ALLOW_SNAPSHOT_ISOLATION OFF 
GO
ALTER DATABASE [Ingestion] SET PARAMETERIZATION SIMPLE 
GO
ALTER DATABASE [Ingestion] SET READ_COMMITTED_SNAPSHOT OFF 
GO
ALTER DATABASE [Ingestion] SET HONOR_BROKER_PRIORITY OFF 
GO
ALTER DATABASE [Ingestion] SET RECOVERY FULL 
GO
ALTER DATABASE [Ingestion] SET  MULTI_USER 
GO
ALTER DATABASE [Ingestion] SET PAGE_VERIFY CHECKSUM  
GO
ALTER DATABASE [Ingestion] SET DB_CHAINING OFF 
GO
ALTER DATABASE [Ingestion] SET FILESTREAM( NON_TRANSACTED_ACCESS = OFF ) 
GO
ALTER DATABASE [Ingestion] SET TARGET_RECOVERY_TIME = 60 SECONDS 
GO
ALTER DATABASE [Ingestion] SET DELAYED_DURABILITY = DISABLED 
GO
ALTER DATABASE [Ingestion] SET ACCELERATED_DATABASE_RECOVERY = OFF  
GO
EXEC sys.sp_db_vardecimal_storage_format N'Ingestion', N'ON'
GO
ALTER DATABASE [Ingestion] SET QUERY_STORE = OFF
GO
USE [Ingestion]
GO
/****** Object:  User [user_tomek]    Script Date: 12.01.2024 16:44:08 ******/
CREATE USER [user_tomek] FOR LOGIN [user_tomek] WITH DEFAULT_SCHEMA=[dbo]
GO
/****** Object:  User [user_anna]    Script Date: 12.01.2024 16:44:08 ******/
CREATE USER [user_anna] FOR LOGIN [user_anna] WITH DEFAULT_SCHEMA=[dbo]
GO
/****** Object:  User [app_superuser]    Script Date: 12.01.2024 16:44:08 ******/
CREATE USER [app_superuser] FOR LOGIN [app_superuser] WITH DEFAULT_SCHEMA=[dbo]
GO
/****** Object:  User [app_restaurants]    Script Date: 12.01.2024 16:44:08 ******/
CREATE USER [app_restaurants] FOR LOGIN [app_restaurants] WITH DEFAULT_SCHEMA=[dbo]
GO
/****** Object:  User [app_reports]    Script Date: 12.01.2024 16:44:08 ******/
CREATE USER [app_reports] FOR LOGIN [app_reports] WITH DEFAULT_SCHEMA=[dbo]
GO
/****** Object:  User [app_orders]    Script Date: 12.01.2024 16:44:08 ******/
CREATE USER [app_orders] FOR LOGIN [app_orders] WITH DEFAULT_SCHEMA=[dbo]
GO
/****** Object:  User [app_admin]    Script Date: 12.01.2024 16:44:08 ******/
CREATE USER [app_admin] FOR LOGIN [app_admin] WITH DEFAULT_SCHEMA=[dbo]
GO
/****** Object:  DatabaseRole [user_maintenance_role]    Script Date: 12.01.2024 16:44:08 ******/
CREATE ROLE [user_maintenance_role]
GO
/****** Object:  DatabaseRole [user_developer_role]    Script Date: 12.01.2024 16:44:08 ******/
CREATE ROLE [user_developer_role]
GO
/****** Object:  DatabaseRole [app_orders_role]    Script Date: 12.01.2024 16:44:08 ******/
CREATE ROLE [app_orders_role]
GO
ALTER ROLE [user_developer_role] ADD MEMBER [user_tomek]
GO
ALTER ROLE [db_datareader] ADD MEMBER [user_tomek]
GO
ALTER ROLE [db_datareader] ADD MEMBER [user_anna]
GO
ALTER ROLE [db_datawriter] ADD MEMBER [user_anna]
GO
ALTER ROLE [db_ddladmin] ADD MEMBER [app_superuser]
GO
ALTER ROLE [db_datareader] ADD MEMBER [app_superuser]
GO
ALTER ROLE [db_datawriter] ADD MEMBER [app_superuser]
GO
ALTER ROLE [db_datareader] ADD MEMBER [app_restaurants]
GO
ALTER ROLE [db_datawriter] ADD MEMBER [app_restaurants]
GO
ALTER ROLE [db_datareader] ADD MEMBER [app_reports]
GO
ALTER ROLE [app_orders_role] ADD MEMBER [app_orders]
GO
ALTER ROLE [db_datareader] ADD MEMBER [app_orders]
GO
ALTER ROLE [db_datareader] ADD MEMBER [app_admin]
GO
ALTER ROLE [db_datawriter] ADD MEMBER [app_admin]
GO
/****** Object:  Schema [Clients]    Script Date: 12.01.2024 16:44:08 ******/
CREATE SCHEMA [Clients]
GO
/****** Object:  Schema [GlobalConfig]    Script Date: 12.01.2024 16:44:08 ******/
CREATE SCHEMA [GlobalConfig]
GO
/****** Object:  Schema [INGESTION]    Script Date: 12.01.2024 16:44:08 ******/
CREATE SCHEMA [INGESTION]
GO
/****** Object:  Schema [Orders]    Script Date: 12.01.2024 16:44:08 ******/
CREATE SCHEMA [Orders]
GO
/****** Object:  Schema [Resources]    Script Date: 12.01.2024 16:44:08 ******/
CREATE SCHEMA [Resources]
GO
/****** Object:  Schema [Restaurants]    Script Date: 12.01.2024 16:44:08 ******/
CREATE SCHEMA [Restaurants]
GO
/****** Object:  Schema [Staff]    Script Date: 12.01.2024 16:44:08 ******/
CREATE SCHEMA [Staff]
GO
/****** Object:  Rule [TITLE_RULE]    Script Date: 12.01.2024 16:44:08 ******/
CREATE RULE [dbo].[TITLE_RULE] 
AS
@title IN ('Mr', 'Mrs', 'Miss');
GO
/****** Object:  UserDefinedFunction [dbo].[GetLinkedTables]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE FUNCTION [dbo].[GetLinkedTables] 
(
    @TableName NVARCHAR(128),
    @SchemaName NVARCHAR(128)
)
RETURNS @Result TABLE 
(
    TableId NVARCHAR(256),
	SchemaName NVARCHAR(256),
    TableName NVARCHAR(256),
	ParentTableName NVARCHAR(256),
	ColumnName NVARCHAR(256),
	Level INT
)
AS
BEGIN
	WITH RecursiveFKs(ForeignKeyName, ParentObjectId, ReferenceObjectId, ParentColumnName, ReferencedColumnName, Level)
		AS
		(
			SELECT 
				FK.name AS ForeignKeyName,
				FK.parent_object_id AS ParentObjectId,
				FK.referenced_object_id AS ReferenceObjectId,
				COL_NAME(FKC.parent_object_id, FKC.parent_column_id) AS ParentColumnName,
				COL_NAME(FKC.referenced_object_id, FKC.referenced_column_id) AS ReferencedColumnName,
				1 AS Level
			FROM 
				sys.foreign_keys AS FK
			INNER JOIN	sys.foreign_key_columns AS FKC
				ON FK.object_id = FKC.constraint_object_id
			WHERE 
				OBJECT_NAME(FK.referenced_object_id) = @TableName 
				AND OBJECT_SCHEMA_NAME(FK.parent_object_id) = @SchemaName

			UNION ALL

			SELECT 
				FK.name AS ForeignKeyName,
				FK.parent_object_id AS ParentObjectId,
				FK.referenced_object_id AS ReferenceObjectId,
				COL_NAME(FKC.parent_object_id, FKC.parent_column_id) AS ParentColumnName,
				COL_NAME(FKC.referenced_object_id, FKC.referenced_column_id) AS ReferencedColumnName,
				R.Level + 1
			FROM 
				sys.foreign_keys AS FK
			INNER JOIN RecursiveFKs AS R
				ON FK.referenced_object_id = R.ParentObjectId
			INNER JOIN	sys.foreign_key_columns AS FKC
				ON FK.object_id = FKC.constraint_object_id
			WHERE
				OBJECT_NAME(fk.referenced_object_id) != @TableName
				AND OBJECT_SCHEMA_NAME(FK.parent_object_id) = @SchemaName
		)
	INSERT INTO @Result
	SELECT 
	DISTINCT 
		ParentObjectId AS TableId,
		OBJECT_SCHEMA_NAME(ParentObjectId) AS SchemaName,
		OBJECT_NAME(ParentObjectId) AS TableName,
		OBJECT_NAME(ReferenceObjectId) AS ParentTableName,
		ParentColumnName AS ColumnName,
		Level
	FROM RecursiveFKs
	ORDER BY Level
    RETURN;
END;
GO
/****** Object:  Table [Restaurants].[Locations]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Restaurants].[Locations](
	[ID] [smallint] IDENTITY(1,1) NOT NULL,
	[Name] [char](255) NOT NULL,
	[Slug] [varchar](255) NOT NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Restaurants].[Addresses]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Restaurants].[Addresses](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[LocationID] [smallint] NOT NULL,
	[AddressOne] [varchar](255) NOT NULL,
	[AddressTwo] [varchar](255) NULL,
	[Longitude] [decimal](9, 6) NOT NULL,
	[Latitude] [decimal](8, 6) NOT NULL,
	[Geom]  AS ([geography]::Point([Latitude],[Longitude],(4326))) PERSISTED,
	[ZipCode] [char](5) NOT NULL,
	[CityId] [int] NOT NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
	[IsCurrent] [bit] NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Restaurants].[PhoneNumbers]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Restaurants].[PhoneNumbers](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[LocationID] [smallint] NOT NULL,
	[Number] [varchar](15) NOT NULL,
	[Description] [varchar](255) NOT NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  View [Restaurants].[LocationDetails]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE VIEW [Restaurants].[LocationDetails] AS
WITH RestaurantPhoneNumbers AS (
	SELECT 
		LocationID
		, Number
		, row_number() over (partition by LocationID order by CreatedAt desc) as RowNum
	FROM
		Restaurants.PhoneNumbers
)
SELECT
	RL.ID AS LocationId
	, RL.Name AS LocationName
	, RA.AddressOne + ' ' + RA.AddressTwo AS LocationAddress
	, RA.ZipCode AS LocationZipCode
	, RA.Geom AS AddressGeom
FROM 
	Restaurants.Locations AS RL
INNER JOIN
	Restaurants.Addresses AS RA
	ON RL.Id = RA.LocationId AND RA.IsCurrent = 1
INNER JOIN 
	RestaurantPhoneNumbers AS RPN
	ON RPN.LocationID = RL.ID AND RPN.RowNum = 1;
GO
/****** Object:  Table [Staff].[Employees]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Staff].[Employees](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[FirstName] [varchar](255) NOT NULL,
	[LastName] [varchar](255) NOT NULL,
	[BirthDate] [date] NOT NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
	[Title] [nvarchar](5) NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Staff].[Roles]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Staff].[Roles](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[Name] [varchar](255) NOT NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Staff].[EmployeeLocationRoles]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Staff].[EmployeeLocationRoles](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[EmployeeID] [int] NOT NULL,
	[LocationID] [smallint] NOT NULL,
	[RoleID] [int] NOT NULL,
	[ValidFrom] [date] NOT NULL,
	[ValidTo] [date] NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  View [Restaurants].[ActiveEmployees]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE VIEW [Restaurants].[ActiveEmployees] AS
SELECT
	SE.ID AS EmployeeId
	, SE.FirstName + ' ' + SE.LastName AS EmployeeFullName
	, SR.Name AS Role
	, SELR.ValidTo AS WorkingUntill
	, SELR.LocationId AS LocationId
FROM Staff.Employees AS SE
INNER JOIN Staff.EmployeeLocationRoles AS SELR
	ON SE.Id = SELR.EmployeeId
INNER JOIN Staff.Roles AS SR
	ON SELR.RoleId = SR.Id
WHERE 
	SELR.ValidFrom <= CAST(GETDATE() AS Date) AND SELR.ValidTo >= CAST(GETDATE() AS Date);
GO
/****** Object:  Table [Clients].[Addresses]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Clients].[Addresses](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[CustomerID] [int] NOT NULL,
	[AddressOne] [varchar](255) NOT NULL,
	[AddressTwo] [varchar](255) NULL,
	[ZipCode] [char](5) NOT NULL,
	[CityID] [int] NOT NULL,
	[Description] [varchar](max) NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
	[DeletedAt] [datetime] NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [Clients].[Customers]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Clients].[Customers](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[FirstName] [varchar](50) NOT NULL,
	[LastName] [varchar](50) NULL,
	[PhoneNumber] [varchar](15) NULL,
	[Email] [varchar](255) NULL,
	[Blocked] [bit] NOT NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
	[Title] [nvarchar](5) NULL,
	[DeletedAt] [datetime] NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [GlobalConfig].[Cities]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [GlobalConfig].[Cities](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[StateId] [int] NOT NULL,
	[Name] [varchar](255) NOT NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [GlobalConfig].[States]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [GlobalConfig].[States](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[Name] [varchar](255) NOT NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Orders].[Deliveries]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Orders].[Deliveries](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[OrderID] [int] NOT NULL,
	[EmployeeID] [int] NOT NULL,
	[Status] [varchar](25) NOT NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
	[DeletedAt] [datetime2](7) NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Orders].[DeliveryItems]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Orders].[DeliveryItems](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[DeliveryID] [int] NOT NULL,
	[OrderItemID] [int] NOT NULL,
	[Quantity] [smallint] NOT NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
	[DeletedAt] [datetime2](7) NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Orders].[DeliveryStatuses]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Orders].[DeliveryStatuses](
	[Status] [varchar](25) NOT NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[Status] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Orders].[DeliveryTimes]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Orders].[DeliveryTimes](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[DeliveryID] [int] NOT NULL,
	[DeliveredTime] [datetime] NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
	[DeletedAt] [datetime2](7) NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Orders].[OrderItems]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Orders].[OrderItems](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[OrderID] [int] NOT NULL,
	[DishID] [int] NOT NULL,
	[Quantity] [smallint] NOT NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
	[DeletedAt] [datetime2](7) NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Orders].[Orders]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Orders].[Orders](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[LocationID] [smallint] NOT NULL,
	[Status] [varchar](25) NOT NULL,
	[DeliveryAddressID] [int] NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
	[DeletedAt] [datetime2](7) NULL,
	[OrderType] [varchar](25) NOT NULL,
	[RequestedDateTime] [datetime] NULL,
	[CustomerID] [int] NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Orders].[OrderStatuses]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Orders].[OrderStatuses](
	[Status] [varchar](25) NOT NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[Status] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Orders].[OrderTypes]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Orders].[OrderTypes](
	[Type] [varchar](25) NOT NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[Type] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Resources].[Allergens]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Resources].[Allergens](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[Name] [varchar](255) NOT NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Resources].[Categories]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Resources].[Categories](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[Name] [varchar](50) NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Resources].[Dishes]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Resources].[Dishes](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[Name] [varchar](100) NOT NULL,
	[Description] [varchar](max) NOT NULL,
	[Price] [smallmoney] NOT NULL,
	[CategoryID] [int] NOT NULL,
	[ChefID] [int] NOT NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
	[DishType] [varchar](25) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [Resources].[DishProducts]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Resources].[DishProducts](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[DishID] [int] NOT NULL,
	[ProductID] [int] NOT NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Resources].[DishTypes]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Resources].[DishTypes](
	[Type] [varchar](25) NOT NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[Type] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Resources].[MenuDishes]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Resources].[MenuDishes](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[MenuID] [int] NOT NULL,
	[DishID] [int] NOT NULL,
	[ValidFrom] [date] NOT NULL,
	[ValidTo] [date] NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Resources].[Menus]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Resources].[Menus](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[Name] [varchar](50) NULL,
	[ValidFrom] [date] NOT NULL,
	[ValidTo] [date] NULL,
	[ExecutiveChefID] [int] NOT NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Resources].[ProductAllergens]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Resources].[ProductAllergens](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[ProductID] [int] NOT NULL,
	[AllergenID] [int] NOT NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Resources].[Products]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Resources].[Products](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[Name] [varchar](255) NOT NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Restaurants].[LocationMenus]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Restaurants].[LocationMenus](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[LocationID] [smallint] NOT NULL,
	[MenuID] [int] NOT NULL,
	[ValidFrom] [date] NOT NULL,
	[ValidTo] [date] NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Restaurants].[OpeningHours]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Restaurants].[OpeningHours](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[ScheduleId] [int] NOT NULL,
	[StartTime] [time](7) NOT NULL,
	[EndTime] [time](7) NOT NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Restaurants].[Reservations]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Restaurants].[Reservations](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[CustomerID] [int] NOT NULL,
	[ReservationDate] [date] NOT NULL,
	[ReservationHour] [time](7) NOT NULL,
	[ReservationTime] [smallint] NOT NULL,
	[Seats] [smallint] NOT NULL,
	[Notes] [varchar](max) NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
	[DeletedAt] [datetime2](7) NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [Restaurants].[Schedules]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Restaurants].[Schedules](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[LocationId] [smallint] NOT NULL,
	[DayOfWeek] [tinyint] NOT NULL,
	[StartDate] [date] NOT NULL,
	[EndDate] [date] NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Restaurants].[TableReservations]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Restaurants].[TableReservations](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[TableID] [int] NOT NULL,
	[ReservationID] [int] NOT NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
	[DeletedAt] [datetime2](7) NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Restaurants].[Tables]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Restaurants].[Tables](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[LocationID] [smallint] NOT NULL,
	[Description] [varchar](50) NOT NULL,
	[Seats] [tinyint] NOT NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
	[DeletedAt] [datetime2](7) NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Staff].[Addresses]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Staff].[Addresses](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[EmployeeID] [int] NOT NULL,
	[AddressOne] [varchar](255) NOT NULL,
	[AddressTwo] [varchar](255) NULL,
	[ZipCode] [char](5) NOT NULL,
	[CityId] [int] NOT NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Staff].[PhoneNumbers]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Staff].[PhoneNumbers](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[NumberType] [varchar](15) NOT NULL,
	[EmployeeID] [int] NOT NULL,
	[Number] [varchar](15) NOT NULL,
	[Description] [varchar](255) NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Staff].[PhoneNumberTypes]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Staff].[PhoneNumberTypes](
	[Type] [varchar](15) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[Type] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Staff].[Schedules]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Staff].[Schedules](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[EmployeeId] [int] NOT NULL,
	[LocationID] [smallint] NOT NULL,
	[DayOfWeek] [tinyint] NOT NULL,
	[RepeatFrequency] [tinyint] NOT NULL,
	[StartDate] [date] NOT NULL,
	[EndDate] [date] NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Staff].[WorkingHours]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Staff].[WorkingHours](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[ScheduleId] [int] NOT NULL,
	[StartTime] [time](7) NOT NULL,
	[EndTime] [time](7) NOT NULL,
	[CreatedAt] [datetime2](7) NOT NULL,
	[UpdatedAt] [datetime2](7) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Index [ClientsAddressesCityIdIdx]    Script Date: 12.01.2024 16:44:08 ******/
CREATE NONCLUSTERED INDEX [ClientsAddressesCityIdIdx] ON [Clients].[Addresses]
(
	[CityID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [ClientsAddressesCustomerIdIdx]    Script Date: 12.01.2024 16:44:08 ******/
CREATE NONCLUSTERED INDEX [ClientsAddressesCustomerIdIdx] ON [Clients].[Addresses]
(
	[CustomerID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [OrdersOrdersDeliveryAddressId]    Script Date: 12.01.2024 16:44:08 ******/
CREATE NONCLUSTERED INDEX [OrdersOrdersDeliveryAddressId] ON [Orders].[Orders]
(
	[DeliveryAddressID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [OrdersOrdersLocationIdIdx]    Script Date: 12.01.2024 16:44:08 ******/
CREATE NONCLUSTERED INDEX [OrdersOrdersLocationIdIdx] ON [Orders].[Orders]
(
	[LocationID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
SET ANSI_PADDING ON
GO
/****** Object:  Index [OrdersOrdersStatusIdx]    Script Date: 12.01.2024 16:44:08 ******/
CREATE NONCLUSTERED INDEX [OrdersOrdersStatusIdx] ON [Orders].[Orders]
(
	[Status] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
SET ANSI_PADDING ON
GO
/****** Object:  Index [OrdersOrdersTypesIdx]    Script Date: 12.01.2024 16:44:08 ******/
CREATE NONCLUSTERED INDEX [OrdersOrdersTypesIdx] ON [Orders].[Orders]
(
	[OrderType] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [ResourcesDishesCategoryIdx]    Script Date: 12.01.2024 16:44:08 ******/
CREATE NONCLUSTERED INDEX [ResourcesDishesCategoryIdx] ON [Resources].[Dishes]
(
	[CategoryID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [ResourcesDishesChefId]    Script Date: 12.01.2024 16:44:08 ******/
CREATE NONCLUSTERED INDEX [ResourcesDishesChefId] ON [Resources].[Dishes]
(
	[ChefID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
SET ANSI_PADDING ON
GO
/****** Object:  Index [ResourcesDishesDishTypeIdx]    Script Date: 12.01.2024 16:44:08 ******/
CREATE NONCLUSTERED INDEX [ResourcesDishesDishTypeIdx] ON [Resources].[Dishes]
(
	[DishType] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [ResourcesMenuDishesDishId]    Script Date: 12.01.2024 16:44:08 ******/
CREATE NONCLUSTERED INDEX [ResourcesMenuDishesDishId] ON [Resources].[MenuDishes]
(
	[DishID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [ResourcesMenuDishesMenuId]    Script Date: 12.01.2024 16:44:08 ******/
CREATE NONCLUSTERED INDEX [ResourcesMenuDishesMenuId] ON [Resources].[MenuDishes]
(
	[MenuID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [ResourcesMenusExecutiveChefId]    Script Date: 12.01.2024 16:44:08 ******/
CREATE NONCLUSTERED INDEX [ResourcesMenusExecutiveChefId] ON [Resources].[Menus]
(
	[ExecutiveChefID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [Restaurants_Current_Address]    Script Date: 12.01.2024 16:44:08 ******/
CREATE UNIQUE NONCLUSTERED INDEX [Restaurants_Current_Address] ON [Restaurants].[Addresses]
(
	[LocationID] ASC,
	[IsCurrent] ASC
)
WHERE ([IsCurrent]=(1))
WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, IGNORE_DUP_KEY = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [RestaurantsAddressesCityIdx]    Script Date: 12.01.2024 16:44:08 ******/
CREATE NONCLUSTERED INDEX [RestaurantsAddressesCityIdx] ON [Restaurants].[Addresses]
(
	[CityId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [RestaurantsAddressesLocationIdx]    Script Date: 12.01.2024 16:44:08 ******/
CREATE NONCLUSTERED INDEX [RestaurantsAddressesLocationIdx] ON [Restaurants].[Addresses]
(
	[LocationID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [RestaurantsOpeningHoursSchedulesIdx]    Script Date: 12.01.2024 16:44:08 ******/
CREATE NONCLUSTERED INDEX [RestaurantsOpeningHoursSchedulesIdx] ON [Restaurants].[OpeningHours]
(
	[ScheduleId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [RestaurantsPhoneNumbersLocationIdx]    Script Date: 12.01.2024 16:44:08 ******/
CREATE NONCLUSTERED INDEX [RestaurantsPhoneNumbersLocationIdx] ON [Restaurants].[PhoneNumbers]
(
	[LocationID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [RestaurantsReservationsCustomerIdIdx]    Script Date: 12.01.2024 16:44:08 ******/
CREATE NONCLUSTERED INDEX [RestaurantsReservationsCustomerIdIdx] ON [Restaurants].[Reservations]
(
	[CustomerID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [RestaurantsSchedulesLocationIdx]    Script Date: 12.01.2024 16:44:08 ******/
CREATE NONCLUSTERED INDEX [RestaurantsSchedulesLocationIdx] ON [Restaurants].[Schedules]
(
	[LocationId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [RestaurantsTableReservationsReservationIdIdx]    Script Date: 12.01.2024 16:44:08 ******/
CREATE NONCLUSTERED INDEX [RestaurantsTableReservationsReservationIdIdx] ON [Restaurants].[TableReservations]
(
	[ReservationID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [RestaurantsTableReservationsTableIdIdx]    Script Date: 12.01.2024 16:44:08 ******/
CREATE NONCLUSTERED INDEX [RestaurantsTableReservationsTableIdIdx] ON [Restaurants].[TableReservations]
(
	[TableID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [RestaurantsTablesLocationIdIdx]    Script Date: 12.01.2024 16:44:08 ******/
CREATE NONCLUSTERED INDEX [RestaurantsTablesLocationIdIdx] ON [Restaurants].[Tables]
(
	[LocationID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [StaffAddressesEmployees]    Script Date: 12.01.2024 16:44:08 ******/
CREATE NONCLUSTERED INDEX [StaffAddressesEmployees] ON [Staff].[Addresses]
(
	[EmployeeID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [StaffAssignmentDatesIdx]    Script Date: 12.01.2024 16:44:08 ******/
CREATE NONCLUSTERED INDEX [StaffAssignmentDatesIdx] ON [Staff].[EmployeeLocationRoles]
(
	[ValidFrom] ASC,
	[ValidTo] DESC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [StaffEmployeeLocationRolesEmployee]    Script Date: 12.01.2024 16:44:08 ******/
CREATE NONCLUSTERED INDEX [StaffEmployeeLocationRolesEmployee] ON [Staff].[EmployeeLocationRoles]
(
	[EmployeeID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [StaffEmployeeLocationRolesLocation]    Script Date: 12.01.2024 16:44:08 ******/
CREATE NONCLUSTERED INDEX [StaffEmployeeLocationRolesLocation] ON [Staff].[EmployeeLocationRoles]
(
	[LocationID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [StaffEmployeeLocationRolesRoles]    Script Date: 12.01.2024 16:44:08 ******/
CREATE NONCLUSTERED INDEX [StaffEmployeeLocationRolesRoles] ON [Staff].[EmployeeLocationRoles]
(
	[RoleID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
SET ANSI_PADDING ON
GO
/****** Object:  Index [StaffPhoneNumbersNumberType]    Script Date: 12.01.2024 16:44:08 ******/
CREATE NONCLUSTERED INDEX [StaffPhoneNumbersNumberType] ON [Staff].[PhoneNumbers]
(
	[NumberType] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [StaffSchedulesDates]    Script Date: 12.01.2024 16:44:08 ******/
CREATE NONCLUSTERED INDEX [StaffSchedulesDates] ON [Staff].[Schedules]
(
	[StartDate] ASC,
	[EndDate] DESC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
ALTER TABLE [Clients].[Addresses] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [Clients].[Addresses] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [Clients].[Customers] ADD  DEFAULT ((0)) FOR [Blocked]
GO
ALTER TABLE [Clients].[Customers] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [Clients].[Customers] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [GlobalConfig].[Cities] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [GlobalConfig].[Cities] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [GlobalConfig].[States] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [GlobalConfig].[States] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [Orders].[Deliveries] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [Orders].[Deliveries] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [Orders].[DeliveryItems] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [Orders].[DeliveryItems] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [Orders].[DeliveryStatuses] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [Orders].[DeliveryStatuses] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [Orders].[DeliveryTimes] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [Orders].[DeliveryTimes] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [Orders].[OrderItems] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [Orders].[OrderItems] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [Orders].[Orders] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [Orders].[Orders] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [Orders].[OrderStatuses] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [Orders].[OrderStatuses] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [Orders].[OrderTypes] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [Orders].[OrderTypes] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [Resources].[Allergens] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [Resources].[Allergens] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [Resources].[Categories] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [Resources].[Categories] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [Resources].[Dishes] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [Resources].[Dishes] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [Resources].[DishProducts] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [Resources].[DishProducts] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [Resources].[DishTypes] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [Resources].[DishTypes] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [Resources].[MenuDishes] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [Resources].[MenuDishes] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [Resources].[Menus] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [Resources].[Menus] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [Resources].[ProductAllergens] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [Resources].[ProductAllergens] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [Resources].[Products] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [Resources].[Products] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [Restaurants].[Addresses] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [Restaurants].[Addresses] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [Restaurants].[Addresses] ADD  DEFAULT ((0)) FOR [IsCurrent]
GO
ALTER TABLE [Restaurants].[LocationMenus] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [Restaurants].[LocationMenus] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [Restaurants].[Locations] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [Restaurants].[Locations] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [Restaurants].[OpeningHours] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [Restaurants].[OpeningHours] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [Restaurants].[PhoneNumbers] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [Restaurants].[PhoneNumbers] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [Restaurants].[Reservations] ADD  DEFAULT ((3600)) FOR [ReservationTime]
GO
ALTER TABLE [Restaurants].[Reservations] ADD  DEFAULT ((1)) FOR [Seats]
GO
ALTER TABLE [Restaurants].[Reservations] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [Restaurants].[Reservations] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [Restaurants].[Schedules] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [Restaurants].[Schedules] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [Restaurants].[TableReservations] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [Restaurants].[TableReservations] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [Restaurants].[Tables] ADD  DEFAULT ((0)) FOR [Seats]
GO
ALTER TABLE [Restaurants].[Tables] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [Restaurants].[Tables] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [Staff].[Addresses] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [Staff].[Addresses] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [Staff].[EmployeeLocationRoles] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [Staff].[EmployeeLocationRoles] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [Staff].[Employees] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [Staff].[Employees] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [Staff].[PhoneNumbers] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [Staff].[PhoneNumbers] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [Staff].[Roles] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [Staff].[Roles] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [Staff].[Schedules] ADD  DEFAULT ((1)) FOR [RepeatFrequency]
GO
ALTER TABLE [Staff].[Schedules] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [Staff].[Schedules] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [Staff].[WorkingHours] ADD  DEFAULT (getdate()) FOR [CreatedAt]
GO
ALTER TABLE [Staff].[WorkingHours] ADD  DEFAULT (getdate()) FOR [UpdatedAt]
GO
ALTER TABLE [Clients].[Addresses]  WITH CHECK ADD  CONSTRAINT [FK_Clients_Addresses_Cities] FOREIGN KEY([CityID])
REFERENCES [GlobalConfig].[Cities] ([ID])
GO
ALTER TABLE [Clients].[Addresses] CHECK CONSTRAINT [FK_Clients_Addresses_Cities]
GO
ALTER TABLE [Clients].[Addresses]  WITH CHECK ADD  CONSTRAINT [FK_Clients_Addresses_Customers] FOREIGN KEY([CustomerID])
REFERENCES [Clients].[Customers] ([ID])
GO
ALTER TABLE [Clients].[Addresses] CHECK CONSTRAINT [FK_Clients_Addresses_Customers]
GO
ALTER TABLE [GlobalConfig].[Cities]  WITH CHECK ADD  CONSTRAINT [FK_GlobalConfig_Cities_States] FOREIGN KEY([StateId])
REFERENCES [GlobalConfig].[States] ([ID])
GO
ALTER TABLE [GlobalConfig].[Cities] CHECK CONSTRAINT [FK_GlobalConfig_Cities_States]
GO
ALTER TABLE [Orders].[Deliveries]  WITH CHECK ADD  CONSTRAINT [FK_Orders_Deliveries_DeliveryStatuses] FOREIGN KEY([Status])
REFERENCES [Orders].[DeliveryStatuses] ([Status])
GO
ALTER TABLE [Orders].[Deliveries] CHECK CONSTRAINT [FK_Orders_Deliveries_DeliveryStatuses]
GO
ALTER TABLE [Orders].[Deliveries]  WITH CHECK ADD  CONSTRAINT [FK_Orders_Deliveries_Employees] FOREIGN KEY([EmployeeID])
REFERENCES [Staff].[Employees] ([ID])
GO
ALTER TABLE [Orders].[Deliveries] CHECK CONSTRAINT [FK_Orders_Deliveries_Employees]
GO
ALTER TABLE [Orders].[Deliveries]  WITH CHECK ADD  CONSTRAINT [FK_Orders_Deliveries_Orders] FOREIGN KEY([OrderID])
REFERENCES [Orders].[Orders] ([ID])
GO
ALTER TABLE [Orders].[Deliveries] CHECK CONSTRAINT [FK_Orders_Deliveries_Orders]
GO
ALTER TABLE [Orders].[DeliveryItems]  WITH CHECK ADD  CONSTRAINT [FK_Orders_DeliveryDishes_Deliveries] FOREIGN KEY([DeliveryID])
REFERENCES [Orders].[Deliveries] ([ID])
GO
ALTER TABLE [Orders].[DeliveryItems] CHECK CONSTRAINT [FK_Orders_DeliveryDishes_Deliveries]
GO
ALTER TABLE [Orders].[DeliveryItems]  WITH CHECK ADD  CONSTRAINT [FK_Orders_DeliveryDishes_DishIDes] FOREIGN KEY([OrderItemID])
REFERENCES [Orders].[OrderItems] ([ID])
GO
ALTER TABLE [Orders].[DeliveryItems] CHECK CONSTRAINT [FK_Orders_DeliveryDishes_DishIDes]
GO
ALTER TABLE [Orders].[DeliveryTimes]  WITH CHECK ADD  CONSTRAINT [FK_Orders_DeliveryTimes_Deliveries] FOREIGN KEY([DeliveryID])
REFERENCES [Orders].[Deliveries] ([ID])
GO
ALTER TABLE [Orders].[DeliveryTimes] CHECK CONSTRAINT [FK_Orders_DeliveryTimes_Deliveries]
GO
ALTER TABLE [Orders].[OrderItems]  WITH CHECK ADD  CONSTRAINT [FK_Orders_OrderItems_Dishes] FOREIGN KEY([DishID])
REFERENCES [Resources].[Dishes] ([ID])
GO
ALTER TABLE [Orders].[OrderItems] CHECK CONSTRAINT [FK_Orders_OrderItems_Dishes]
GO
ALTER TABLE [Orders].[OrderItems]  WITH CHECK ADD  CONSTRAINT [FK_Orders_OrderItems_Orders] FOREIGN KEY([OrderID])
REFERENCES [Orders].[Orders] ([ID])
GO
ALTER TABLE [Orders].[OrderItems] CHECK CONSTRAINT [FK_Orders_OrderItems_Orders]
GO
ALTER TABLE [Orders].[Orders]  WITH CHECK ADD  CONSTRAINT [FK_Orders_Orders_Customers] FOREIGN KEY([CustomerID])
REFERENCES [Clients].[Customers] ([ID])
GO
ALTER TABLE [Orders].[Orders] CHECK CONSTRAINT [FK_Orders_Orders_Customers]
GO
ALTER TABLE [Orders].[Orders]  WITH CHECK ADD  CONSTRAINT [FK_Orders_Orders_DeliveryAddresses] FOREIGN KEY([DeliveryAddressID])
REFERENCES [Clients].[Addresses] ([ID])
GO
ALTER TABLE [Orders].[Orders] CHECK CONSTRAINT [FK_Orders_Orders_DeliveryAddresses]
GO
ALTER TABLE [Orders].[Orders]  WITH CHECK ADD  CONSTRAINT [FK_Orders_Orders_Locations] FOREIGN KEY([LocationID])
REFERENCES [Restaurants].[Locations] ([ID])
GO
ALTER TABLE [Orders].[Orders] CHECK CONSTRAINT [FK_Orders_Orders_Locations]
GO
ALTER TABLE [Orders].[Orders]  WITH CHECK ADD  CONSTRAINT [FK_Orders_Orders_OrderStatuses] FOREIGN KEY([Status])
REFERENCES [Orders].[OrderStatuses] ([Status])
GO
ALTER TABLE [Orders].[Orders] CHECK CONSTRAINT [FK_Orders_Orders_OrderStatuses]
GO
ALTER TABLE [Orders].[Orders]  WITH CHECK ADD  CONSTRAINT [FK_Orders_Orders_OrderTypes] FOREIGN KEY([OrderType])
REFERENCES [Orders].[OrderTypes] ([Type])
GO
ALTER TABLE [Orders].[Orders] CHECK CONSTRAINT [FK_Orders_Orders_OrderTypes]
GO
ALTER TABLE [Resources].[Dishes]  WITH CHECK ADD  CONSTRAINT [FK_Resources_Dishes_Categories] FOREIGN KEY([CategoryID])
REFERENCES [Resources].[Categories] ([ID])
GO
ALTER TABLE [Resources].[Dishes] CHECK CONSTRAINT [FK_Resources_Dishes_Categories]
GO
ALTER TABLE [Resources].[Dishes]  WITH CHECK ADD  CONSTRAINT [FK_Resources_Dishes_DishTypes] FOREIGN KEY([DishType])
REFERENCES [Resources].[DishTypes] ([Type])
GO
ALTER TABLE [Resources].[Dishes] CHECK CONSTRAINT [FK_Resources_Dishes_DishTypes]
GO
ALTER TABLE [Resources].[Dishes]  WITH CHECK ADD  CONSTRAINT [FK_Resources_Dishes_Employees] FOREIGN KEY([ChefID])
REFERENCES [Staff].[Employees] ([ID])
GO
ALTER TABLE [Resources].[Dishes] CHECK CONSTRAINT [FK_Resources_Dishes_Employees]
GO
ALTER TABLE [Resources].[DishProducts]  WITH CHECK ADD  CONSTRAINT [FK_Resources_DishProducts_Dish] FOREIGN KEY([DishID])
REFERENCES [Resources].[Dishes] ([ID])
GO
ALTER TABLE [Resources].[DishProducts] CHECK CONSTRAINT [FK_Resources_DishProducts_Dish]
GO
ALTER TABLE [Resources].[DishProducts]  WITH CHECK ADD  CONSTRAINT [FK_Resources_DishProducts_Product] FOREIGN KEY([ProductID])
REFERENCES [Resources].[Products] ([ID])
GO
ALTER TABLE [Resources].[DishProducts] CHECK CONSTRAINT [FK_Resources_DishProducts_Product]
GO
ALTER TABLE [Resources].[MenuDishes]  WITH CHECK ADD  CONSTRAINT [FK_Resources_MenuDishes_Dish] FOREIGN KEY([DishID])
REFERENCES [Resources].[Dishes] ([ID])
GO
ALTER TABLE [Resources].[MenuDishes] CHECK CONSTRAINT [FK_Resources_MenuDishes_Dish]
GO
ALTER TABLE [Resources].[MenuDishes]  WITH CHECK ADD  CONSTRAINT [FK_Resources_MenuDishes_Menu] FOREIGN KEY([MenuID])
REFERENCES [Resources].[Menus] ([ID])
GO
ALTER TABLE [Resources].[MenuDishes] CHECK CONSTRAINT [FK_Resources_MenuDishes_Menu]
GO
ALTER TABLE [Resources].[Menus]  WITH CHECK ADD  CONSTRAINT [FK_Resources_Menu_Employees] FOREIGN KEY([ExecutiveChefID])
REFERENCES [Staff].[Employees] ([ID])
GO
ALTER TABLE [Resources].[Menus] CHECK CONSTRAINT [FK_Resources_Menu_Employees]
GO
ALTER TABLE [Resources].[ProductAllergens]  WITH CHECK ADD  CONSTRAINT [FK_Resources_ProductAllergens_Allergen] FOREIGN KEY([AllergenID])
REFERENCES [Resources].[Allergens] ([ID])
GO
ALTER TABLE [Resources].[ProductAllergens] CHECK CONSTRAINT [FK_Resources_ProductAllergens_Allergen]
GO
ALTER TABLE [Resources].[ProductAllergens]  WITH CHECK ADD  CONSTRAINT [FK_Resources_ProductAllergens_Product] FOREIGN KEY([ProductID])
REFERENCES [Resources].[Products] ([ID])
GO
ALTER TABLE [Resources].[ProductAllergens] CHECK CONSTRAINT [FK_Resources_ProductAllergens_Product]
GO
ALTER TABLE [Restaurants].[Addresses]  WITH CHECK ADD  CONSTRAINT [FK_Restaurants_Addresses_Cities] FOREIGN KEY([CityId])
REFERENCES [GlobalConfig].[Cities] ([ID])
GO
ALTER TABLE [Restaurants].[Addresses] CHECK CONSTRAINT [FK_Restaurants_Addresses_Cities]
GO
ALTER TABLE [Restaurants].[Addresses]  WITH CHECK ADD  CONSTRAINT [FK_Restaurants_Addresses_Locations] FOREIGN KEY([LocationID])
REFERENCES [Restaurants].[Locations] ([ID])
GO
ALTER TABLE [Restaurants].[Addresses] CHECK CONSTRAINT [FK_Restaurants_Addresses_Locations]
GO
ALTER TABLE [Restaurants].[LocationMenus]  WITH CHECK ADD  CONSTRAINT [FK_Restaurants_LocationMenus_Locations] FOREIGN KEY([LocationID])
REFERENCES [Restaurants].[Locations] ([ID])
GO
ALTER TABLE [Restaurants].[LocationMenus] CHECK CONSTRAINT [FK_Restaurants_LocationMenus_Locations]
GO
ALTER TABLE [Restaurants].[LocationMenus]  WITH CHECK ADD  CONSTRAINT [FK_Restaurants_LocationMenus_Menus] FOREIGN KEY([MenuID])
REFERENCES [Resources].[Menus] ([ID])
GO
ALTER TABLE [Restaurants].[LocationMenus] CHECK CONSTRAINT [FK_Restaurants_LocationMenus_Menus]
GO
ALTER TABLE [Restaurants].[OpeningHours]  WITH CHECK ADD  CONSTRAINT [FK_Restaurants_OpeningHours_Schedule] FOREIGN KEY([ScheduleId])
REFERENCES [Restaurants].[Schedules] ([ID])
GO
ALTER TABLE [Restaurants].[OpeningHours] CHECK CONSTRAINT [FK_Restaurants_OpeningHours_Schedule]
GO
ALTER TABLE [Restaurants].[PhoneNumbers]  WITH CHECK ADD  CONSTRAINT [FK_Restaurants_PhoneNumbers_Locations] FOREIGN KEY([LocationID])
REFERENCES [Restaurants].[Locations] ([ID])
GO
ALTER TABLE [Restaurants].[PhoneNumbers] CHECK CONSTRAINT [FK_Restaurants_PhoneNumbers_Locations]
GO
ALTER TABLE [Restaurants].[Reservations]  WITH CHECK ADD  CONSTRAINT [FK_Restaurants_Reservations_Customers] FOREIGN KEY([CustomerID])
REFERENCES [Clients].[Customers] ([ID])
GO
ALTER TABLE [Restaurants].[Reservations] CHECK CONSTRAINT [FK_Restaurants_Reservations_Customers]
GO
ALTER TABLE [Restaurants].[Schedules]  WITH CHECK ADD  CONSTRAINT [FK_Restaurants_Schedules_Locations] FOREIGN KEY([LocationId])
REFERENCES [Restaurants].[Locations] ([ID])
GO
ALTER TABLE [Restaurants].[Schedules] CHECK CONSTRAINT [FK_Restaurants_Schedules_Locations]
GO
ALTER TABLE [Restaurants].[TableReservations]  WITH CHECK ADD  CONSTRAINT [FK_Restaurants_TableReservations_Reservations] FOREIGN KEY([ReservationID])
REFERENCES [Restaurants].[Reservations] ([ID])
GO
ALTER TABLE [Restaurants].[TableReservations] CHECK CONSTRAINT [FK_Restaurants_TableReservations_Reservations]
GO
ALTER TABLE [Restaurants].[TableReservations]  WITH CHECK ADD  CONSTRAINT [FK_Restaurants_TableReservations_Tables] FOREIGN KEY([TableID])
REFERENCES [Restaurants].[Tables] ([ID])
GO
ALTER TABLE [Restaurants].[TableReservations] CHECK CONSTRAINT [FK_Restaurants_TableReservations_Tables]
GO
ALTER TABLE [Restaurants].[Tables]  WITH CHECK ADD  CONSTRAINT [FK_Restaurants_Tables_Locations] FOREIGN KEY([LocationID])
REFERENCES [Restaurants].[Locations] ([ID])
GO
ALTER TABLE [Restaurants].[Tables] CHECK CONSTRAINT [FK_Restaurants_Tables_Locations]
GO
ALTER TABLE [Staff].[Addresses]  WITH CHECK ADD  CONSTRAINT [FK_Staff_Addresses_Cities] FOREIGN KEY([CityId])
REFERENCES [GlobalConfig].[Cities] ([ID])
GO
ALTER TABLE [Staff].[Addresses] CHECK CONSTRAINT [FK_Staff_Addresses_Cities]
GO
ALTER TABLE [Staff].[Addresses]  WITH CHECK ADD  CONSTRAINT [FK_Staff_Addresses_Locations] FOREIGN KEY([EmployeeID])
REFERENCES [Staff].[Employees] ([ID])
GO
ALTER TABLE [Staff].[Addresses] CHECK CONSTRAINT [FK_Staff_Addresses_Locations]
GO
ALTER TABLE [Staff].[EmployeeLocationRoles]  WITH CHECK ADD  CONSTRAINT [FK_Staff_EmployeeLocationRoles_Employees] FOREIGN KEY([EmployeeID])
REFERENCES [Staff].[Employees] ([ID])
GO
ALTER TABLE [Staff].[EmployeeLocationRoles] CHECK CONSTRAINT [FK_Staff_EmployeeLocationRoles_Employees]
GO
ALTER TABLE [Staff].[EmployeeLocationRoles]  WITH CHECK ADD  CONSTRAINT [FK_Staff_EmployeeLocationRoles_Locations] FOREIGN KEY([LocationID])
REFERENCES [Restaurants].[Locations] ([ID])
GO
ALTER TABLE [Staff].[EmployeeLocationRoles] CHECK CONSTRAINT [FK_Staff_EmployeeLocationRoles_Locations]
GO
ALTER TABLE [Staff].[EmployeeLocationRoles]  WITH CHECK ADD  CONSTRAINT [FK_Staff_EmployeeLocationRoles_Roles] FOREIGN KEY([RoleID])
REFERENCES [Staff].[Roles] ([ID])
GO
ALTER TABLE [Staff].[EmployeeLocationRoles] CHECK CONSTRAINT [FK_Staff_EmployeeLocationRoles_Roles]
GO
ALTER TABLE [Staff].[PhoneNumbers]  WITH CHECK ADD  CONSTRAINT [FK_Staff_PhoneNumebers_Locations] FOREIGN KEY([EmployeeID])
REFERENCES [Staff].[Employees] ([ID])
GO
ALTER TABLE [Staff].[PhoneNumbers] CHECK CONSTRAINT [FK_Staff_PhoneNumebers_Locations]
GO
ALTER TABLE [Staff].[PhoneNumbers]  WITH CHECK ADD  CONSTRAINT [FK_Staff_PhoneNumebers_PhoneNumberTypes] FOREIGN KEY([NumberType])
REFERENCES [Staff].[PhoneNumberTypes] ([Type])
GO
ALTER TABLE [Staff].[PhoneNumbers] CHECK CONSTRAINT [FK_Staff_PhoneNumebers_PhoneNumberTypes]
GO
ALTER TABLE [Staff].[Schedules]  WITH CHECK ADD  CONSTRAINT [FK_Staff_Schedules_Employees] FOREIGN KEY([EmployeeId])
REFERENCES [Staff].[Employees] ([ID])
GO
ALTER TABLE [Staff].[Schedules] CHECK CONSTRAINT [FK_Staff_Schedules_Employees]
GO
ALTER TABLE [Staff].[Schedules]  WITH CHECK ADD  CONSTRAINT [FK_Staff_Schedules_Locations] FOREIGN KEY([LocationID])
REFERENCES [Restaurants].[Locations] ([ID])
GO
ALTER TABLE [Staff].[Schedules] CHECK CONSTRAINT [FK_Staff_Schedules_Locations]
GO
ALTER TABLE [Staff].[WorkingHours]  WITH CHECK ADD  CONSTRAINT [FK_Staff_WorkingHours_Schedule] FOREIGN KEY([ScheduleId])
REFERENCES [Staff].[Schedules] ([ID])
GO
ALTER TABLE [Staff].[WorkingHours] CHECK CONSTRAINT [FK_Staff_WorkingHours_Schedule]
GO
ALTER TABLE [Orders].[DeliveryItems]  WITH CHECK ADD  CONSTRAINT [CK_Orders_DeliveryItems_Quantity] CHECK  (([Quantity]>(0)))
GO
ALTER TABLE [Orders].[DeliveryItems] CHECK CONSTRAINT [CK_Orders_DeliveryItems_Quantity]
GO
ALTER TABLE [Orders].[DeliveryTimes]  WITH CHECK ADD  CONSTRAINT [CHK_DeliveryTime_NotFuture] CHECK  (([DeliveredTime]<=getdate()))
GO
ALTER TABLE [Orders].[DeliveryTimes] CHECK CONSTRAINT [CHK_DeliveryTime_NotFuture]
GO
ALTER TABLE [Orders].[OrderItems]  WITH CHECK ADD  CONSTRAINT [CK_Orders_DeliveryDishes_Quantity] CHECK  (([Quantity]>(0)))
GO
ALTER TABLE [Orders].[OrderItems] CHECK CONSTRAINT [CK_Orders_DeliveryDishes_Quantity]
GO
ALTER TABLE [Resources].[MenuDishes]  WITH CHECK ADD  CONSTRAINT [CK_Resources_MenuDishes_Valid] CHECK  (([ValidTo] IS NULL OR [ValidFrom]<=[ValidTo]))
GO
ALTER TABLE [Resources].[MenuDishes] CHECK CONSTRAINT [CK_Resources_MenuDishes_Valid]
GO
ALTER TABLE [Resources].[Menus]  WITH CHECK ADD  CONSTRAINT [CK_Resources_Dishes_Valid] CHECK  (([ValidTo] IS NULL OR [ValidFrom]<=[ValidTo]))
GO
ALTER TABLE [Resources].[Menus] CHECK CONSTRAINT [CK_Resources_Dishes_Valid]
GO
ALTER TABLE [Restaurants].[LocationMenus]  WITH CHECK ADD  CONSTRAINT [CK_Restaurants_LocationMenus_Valid] CHECK  (([ValidTo] IS NULL OR [ValidFrom]<=[ValidTo]))
GO
ALTER TABLE [Restaurants].[LocationMenus] CHECK CONSTRAINT [CK_Restaurants_LocationMenus_Valid]
GO
ALTER TABLE [Restaurants].[OpeningHours]  WITH CHECK ADD  CONSTRAINT [CK_Restaurants_LocationsSchedulesOpeningHours_Time] CHECK  (([StartTime]<=[EndTime]))
GO
ALTER TABLE [Restaurants].[OpeningHours] CHECK CONSTRAINT [CK_Restaurants_LocationsSchedulesOpeningHours_Time]
GO
ALTER TABLE [Restaurants].[Schedules]  WITH CHECK ADD  CONSTRAINT [CK_Restaurants_LocationsSchedules_Date] CHECK  (([EndDate] IS NULL OR [StartDate]<=[EndDate]))
GO
ALTER TABLE [Restaurants].[Schedules] CHECK CONSTRAINT [CK_Restaurants_LocationsSchedules_Date]
GO
ALTER TABLE [Staff].[EmployeeLocationRoles]  WITH CHECK ADD  CONSTRAINT [CK_Staff_EmployeeLocationRoles_ValidFromValidTo] CHECK  (([ValidTo] IS NULL OR [ValidFrom]<=[ValidTo]))
GO
ALTER TABLE [Staff].[EmployeeLocationRoles] CHECK CONSTRAINT [CK_Staff_EmployeeLocationRoles_ValidFromValidTo]
GO
ALTER TABLE [Staff].[Schedules]  WITH CHECK ADD  CONSTRAINT [CK_Staff_Schedules_Date] CHECK  (([EndDate] IS NULL OR [StartDate]<=[EndDate]))
GO
ALTER TABLE [Staff].[Schedules] CHECK CONSTRAINT [CK_Staff_Schedules_Date]
GO
ALTER TABLE [Staff].[Schedules]  WITH CHECK ADD  CONSTRAINT [CK_Staff_Schedules_RepeatFrequency] CHECK  (([RepeatFrequency]>(0) AND [RepeatFrequency]<=(4)))
GO
ALTER TABLE [Staff].[Schedules] CHECK CONSTRAINT [CK_Staff_Schedules_RepeatFrequency]
GO
ALTER TABLE [Staff].[WorkingHours]  WITH CHECK ADD  CONSTRAINT [CK_Staff_WorkingHours_Time] CHECK  (([StartTime]<=[EndTime]))
GO
ALTER TABLE [Staff].[WorkingHours] CHECK CONSTRAINT [CK_Staff_WorkingHours_Time]
GO
/****** Object:  StoredProcedure [dbo].[AddAnonizmiationPropertyToColumn]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE PROCEDURE [dbo].[AddAnonizmiationPropertyToColumn]
    @SchemaName NVARCHAR(128),
    @TableName NVARCHAR(128),
    @ColumnName NVARCHAR(128)
AS
BEGIN
    SET NOCOUNT ON;

    IF EXISTS (
        SELECT * 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = @SchemaName 
        AND TABLE_NAME = @TableName 
        AND COLUMN_NAME = @ColumnName
    )
    BEGIN
		EXEC sp_addextendedproperty 
			@name = N'Anonimization', 
			@value = N'Yes', 
			@level0type = N'Schema', @level0name = @SchemaName, 
			@level1type = N'Table',  @level1name = @TableName, 
			@level2type = N'Column', @level2name = @ColumnName;

		PRINT('Anonimization Propery Added');
    END
    ELSE
    BEGIN
        PRINT('Column Does Not Exists');
    END
END;
GO
/****** Object:  StoredProcedure [dbo].[AddExtendedPropertyToColumn]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE PROCEDURE [dbo].[AddExtendedPropertyToColumn]
    @SchemaName NVARCHAR(128),
    @TableName NVARCHAR(128),
    @ColumnName NVARCHAR(128),
    @PropertyName NVARCHAR(128),
    @PropertyValue NVARCHAR(128),
    @Result NVARCHAR(MAX) OUTPUT
AS
BEGIN
    SET NOCOUNT ON;

    IF EXISTS (
        SELECT * 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = @SchemaName 
        AND TABLE_NAME = @TableName 
        AND COLUMN_NAME = @ColumnName
    )
    BEGIN
        BEGIN TRY
            EXEC sp_addextendedproperty 
                @name = @PropertyName, 
                @value = @PropertyValue, 
                @level0type = N'Schema', @level0name = @SchemaName, 
                @level1type = N'Table',  @level1name = @TableName, 
                @level2type = N'Column', @level2name = @ColumnName;

            SET @Result = 'Anonimization Propery Added';
        END TRY
        BEGIN CATCH
            SET @Result = ERROR_MESSAGE();
        END CATCH
    END
    ELSE
    BEGIN
        SET @Result = 'Column Does Not Exists';
    END
END;
GO
/****** Object:  StoredProcedure [dbo].[AnonimizeData]    Script Date: 12.01.2024 16:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE PROCEDURE [dbo].[AnonimizeData]
    @SchemaName NVARCHAR(128),
    @TableName NVARCHAR(128),
	@RelationColumn NVARCHAR(128),
    @IdValue INT
AS
BEGIN
    SET NOCOUNT ON;

    DECLARE @ColumnName NVARCHAR(128);
    DECLARE @QUERY NVARCHAR(MAX);

    DECLARE cursor_column CURSOR FOR 
		SELECT 
			col.name AS ColumnName
		FROM 
			sys.extended_properties AS SEXT
			INNER JOIN sys.columns AS COL
				ON SEXT.major_id = COL.object_id AND SEXT.minor_id = COL.column_id
			INNER JOIN sys.tables AS TBL
				ON COL.object_id = TBL.object_id
			WHERE 
				TBL.name = @TableName
				AND SCHEMA_NAME(TBL.schema_id) = @SchemaName
				AND SEXT.name = 'Anonimization'
				AND SEXT.Value = 'Yes';

    OPEN cursor_column;
    FETCH NEXT FROM cursor_column INTO @ColumnName;

    WHILE @@FETCH_STATUS = 0
    BEGIN
        PRINT('Processing Column ' + @ColumnName);
        SET @QUERY = 'UPDATE ' + QUOTENAME(@SchemaName) + '.' + QUOTENAME(@TableName) + 
                     ' SET ' + QUOTENAME(@ColumnName) + ' = ''Anonimized''' + 
                     ' WHERE ' + QUOTENAME(@ColumnName) + ' IS NOT NULL AND ' + @RelationColumn + ' = ' + CAST(@IdValue AS NVARCHAR(10)) + ';';
        EXEC sp_executesql @QUERY;

        FETCH NEXT FROM cursor_column INTO @ColumnName;
    END

    CLOSE cursor_column;
    DEALLOCATE cursor_column;
END;
GO
EXEC sys.sp_addextendedproperty @name=N'Anonimization', @value=N'Yes' , @level0type=N'SCHEMA',@level0name=N'Clients', @level1type=N'TABLE',@level1name=N'Addresses', @level2type=N'COLUMN',@level2name=N'AddressOne'
GO
EXEC sys.sp_addextendedproperty @name=N'Anonimization', @value=N'Yes' , @level0type=N'SCHEMA',@level0name=N'Clients', @level1type=N'TABLE',@level1name=N'Addresses', @level2type=N'COLUMN',@level2name=N'AddressTwo'
GO
EXEC sys.sp_addextendedproperty @name=N'Anonimization', @value=N'Yes' , @level0type=N'SCHEMA',@level0name=N'Clients', @level1type=N'TABLE',@level1name=N'Addresses', @level2type=N'COLUMN',@level2name=N'Description'
GO
EXEC sys.sp_addextendedproperty @name=N'Anonimization', @value=N'Yes' , @level0type=N'SCHEMA',@level0name=N'Clients', @level1type=N'TABLE',@level1name=N'Customers', @level2type=N'COLUMN',@level2name=N'FirstName'
GO
EXEC sys.sp_addextendedproperty @name=N'Anonimization', @value=N'Yes' , @level0type=N'SCHEMA',@level0name=N'Clients', @level1type=N'TABLE',@level1name=N'Customers', @level2type=N'COLUMN',@level2name=N'LastName'
GO
EXEC sys.sp_addextendedproperty @name=N'Anonimization', @value=N'Yes' , @level0type=N'SCHEMA',@level0name=N'Clients', @level1type=N'TABLE',@level1name=N'Customers', @level2type=N'COLUMN',@level2name=N'Email'
GO
SET ARITHABORT ON
SET CONCAT_NULL_YIELDS_NULL ON
SET QUOTED_IDENTIFIER ON
SET ANSI_NULLS ON
SET ANSI_PADDING ON
SET ANSI_WARNINGS ON
SET NUMERIC_ROUNDABORT OFF
GO
/****** Object:  Index [SIndx_Restaurants_Addresses_Geom]    Script Date: 12.01.2024 16:44:08 ******/
CREATE SPATIAL INDEX [SIndx_Restaurants_Addresses_Geom] ON [Restaurants].[Addresses]
(
	[Geom]
)USING  GEOGRAPHY_AUTO_GRID 
WITH (
CELLS_PER_OBJECT = 12, PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
GO
USE [master]
GO
ALTER DATABASE [Ingestion] SET  READ_WRITE 
GO
