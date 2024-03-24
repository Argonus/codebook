USE [master]
GO
/****** Object:  Database [Cleaned]    Script Date: 12.01.2024 16:34:54 ******/
CREATE DATABASE [Cleaned]
 CONTAINMENT = NONE
 ON  PRIMARY 
( NAME = N'Cleaned', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL15.MSSQLSERVER\MSSQL\DATA\Cleaned.mdf' , SIZE = 8192KB , MAXSIZE = UNLIMITED, FILEGROWTH = 65536KB )
 LOG ON 
( NAME = N'Cleaned_log', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL15.MSSQLSERVER\MSSQL\DATA\Cleaned_log.ldf' , SIZE = 8192KB , MAXSIZE = 2048GB , FILEGROWTH = 65536KB )
 WITH CATALOG_COLLATION = DATABASE_DEFAULT
GO
ALTER DATABASE [Cleaned] SET COMPATIBILITY_LEVEL = 150
GO
IF (1 = FULLTEXTSERVICEPROPERTY('IsFullTextInstalled'))
begin
EXEC [Cleaned].[dbo].[sp_fulltext_database] @action = 'enable'
end
GO
ALTER DATABASE [Cleaned] SET ANSI_NULL_DEFAULT OFF 
GO
ALTER DATABASE [Cleaned] SET ANSI_NULLS OFF 
GO
ALTER DATABASE [Cleaned] SET ANSI_PADDING OFF 
GO
ALTER DATABASE [Cleaned] SET ANSI_WARNINGS OFF 
GO
ALTER DATABASE [Cleaned] SET ARITHABORT OFF 
GO
ALTER DATABASE [Cleaned] SET AUTO_CLOSE OFF 
GO
ALTER DATABASE [Cleaned] SET AUTO_SHRINK OFF 
GO
ALTER DATABASE [Cleaned] SET AUTO_UPDATE_STATISTICS ON 
GO
ALTER DATABASE [Cleaned] SET CURSOR_CLOSE_ON_COMMIT OFF 
GO
ALTER DATABASE [Cleaned] SET CURSOR_DEFAULT  GLOBAL 
GO
ALTER DATABASE [Cleaned] SET CONCAT_NULL_YIELDS_NULL OFF 
GO
ALTER DATABASE [Cleaned] SET NUMERIC_ROUNDABORT OFF 
GO
ALTER DATABASE [Cleaned] SET QUOTED_IDENTIFIER OFF 
GO
ALTER DATABASE [Cleaned] SET RECURSIVE_TRIGGERS OFF 
GO
ALTER DATABASE [Cleaned] SET  ENABLE_BROKER 
GO
ALTER DATABASE [Cleaned] SET AUTO_UPDATE_STATISTICS_ASYNC OFF 
GO
ALTER DATABASE [Cleaned] SET DATE_CORRELATION_OPTIMIZATION OFF 
GO
ALTER DATABASE [Cleaned] SET TRUSTWORTHY OFF 
GO
ALTER DATABASE [Cleaned] SET ALLOW_SNAPSHOT_ISOLATION OFF 
GO
ALTER DATABASE [Cleaned] SET PARAMETERIZATION SIMPLE 
GO
ALTER DATABASE [Cleaned] SET READ_COMMITTED_SNAPSHOT OFF 
GO
ALTER DATABASE [Cleaned] SET HONOR_BROKER_PRIORITY OFF 
GO
ALTER DATABASE [Cleaned] SET RECOVERY FULL 
GO
ALTER DATABASE [Cleaned] SET  MULTI_USER 
GO
ALTER DATABASE [Cleaned] SET PAGE_VERIFY CHECKSUM  
GO
ALTER DATABASE [Cleaned] SET DB_CHAINING OFF 
GO
ALTER DATABASE [Cleaned] SET FILESTREAM( NON_TRANSACTED_ACCESS = OFF ) 
GO
ALTER DATABASE [Cleaned] SET TARGET_RECOVERY_TIME = 60 SECONDS 
GO
ALTER DATABASE [Cleaned] SET DELAYED_DURABILITY = DISABLED 
GO
ALTER DATABASE [Cleaned] SET ACCELERATED_DATABASE_RECOVERY = OFF  
GO
EXEC sys.sp_db_vardecimal_storage_format N'Cleaned', N'ON'
GO
ALTER DATABASE [Cleaned] SET QUERY_STORE = OFF
GO
USE [Cleaned]
GO
/****** Object:  Schema [Clients]    Script Date: 12.01.2024 16:34:55 ******/
CREATE SCHEMA [Clients]
GO
/****** Object:  Schema [Customers]    Script Date: 12.01.2024 16:34:55 ******/
CREATE SCHEMA [Customers]
GO
/****** Object:  Schema [GlobalConfig]    Script Date: 12.01.2024 16:34:55 ******/
CREATE SCHEMA [GlobalConfig]
GO
/****** Object:  Schema [Orders]    Script Date: 12.01.2024 16:34:55 ******/
CREATE SCHEMA [Orders]
GO
/****** Object:  Schema [Resources]    Script Date: 12.01.2024 16:34:55 ******/
CREATE SCHEMA [Resources]
GO
/****** Object:  Schema [Restaurants]    Script Date: 12.01.2024 16:34:55 ******/
CREATE SCHEMA [Restaurants]
GO
/****** Object:  Schema [Staff]    Script Date: 12.01.2024 16:34:55 ******/
CREATE SCHEMA [Staff]
GO
/****** Object:  View [Clients].[CLN_Addresses]    Script Date: 12.01.2024 16:34:55 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE VIEW [Clients].[CLN_Addresses] AS (
	SELECT 
		*
	FROM
		[Ingestion].Clients.Addresses
)
GO
/****** Object:  View [Clients].[CLN_Customers]    Script Date: 12.01.2024 16:34:55 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE VIEW [Clients].[CLN_Customers] AS (
	SELECT 
		*
	FROM
		[Ingestion].Clients.Customers
)
GO
/****** Object:  View [GlobalConfig].[CLN_Cities]    Script Date: 12.01.2024 16:34:55 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE VIEW [GlobalConfig].[CLN_Cities] AS (
	SELECT
		*
	FROM
		[Ingestion].GlobalConfig.Cities
)
GO
/****** Object:  View [GlobalConfig].[CLN_States]    Script Date: 12.01.2024 16:34:55 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE VIEW [GlobalConfig].[CLN_States] AS (
	SELECT
		*
	FROM
		[Ingestion].GlobalConfig.States
)
GO
/****** Object:  View [Orders].[CLN_Deliveries]    Script Date: 12.01.2024 16:34:55 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE VIEW [Orders].[CLN_Deliveries] AS (
	SELECT 
		*
	FROM
		[Ingestion].[Orders].[Deliveries]
	WHERE
		DeletedAt IS NULL
)
GO
/****** Object:  View [Orders].[CLN_DeliveryItems]    Script Date: 12.01.2024 16:34:55 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE VIEW [Orders].[CLN_DeliveryItems] AS (
	SELECT 
		*
	FROM
		[Ingestion].[Orders].DeliveryItems
	WHERE
		DeletedAt IS NULL
)
GO
/****** Object:  View [Orders].[CLN_DeliveryStatuses]    Script Date: 12.01.2024 16:34:55 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE VIEW [Orders].[CLN_DeliveryStatuses] AS (
	SELECT 
		*
	FROM
		[Ingestion].[Orders].DeliveryStatuses
)
GO
/****** Object:  View [Orders].[CLN_DeliveryTimes]    Script Date: 12.01.2024 16:34:55 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE VIEW [Orders].[CLN_DeliveryTimes] AS (
	SELECT 
		*
	FROM
		[Ingestion].[Orders].DeliveryTimes
	WHERE
		DeletedAt IS NULL
)
GO
/****** Object:  View [Orders].[CLN_OrderItems]    Script Date: 12.01.2024 16:34:55 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE VIEW [Orders].[CLN_OrderItems] AS (
	SELECT 
		*
	FROM
		[Ingestion].[Orders].[OrderItems]
	WHERE
		DeletedAt IS NULL
)
GO
/****** Object:  View [Orders].[CLN_Orders]    Script Date: 12.01.2024 16:34:55 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE VIEW [Orders].[CLN_Orders] AS (
	SELECT 
		*
	FROM
		[Ingestion].[Orders].[Orders]
	WHERE
		DeletedAt IS NULL
)
GO
/****** Object:  View [Orders].[CLN_OrderStatuses]    Script Date: 12.01.2024 16:34:55 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE VIEW [Orders].[CLN_OrderStatuses] AS (
	SELECT 
		*
	FROM
		[Ingestion].[Orders].[OrderStatuses]
)
GO
/****** Object:  View [Orders].[CLN_OrderTypes]    Script Date: 12.01.2024 16:34:55 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE VIEW [Orders].[CLN_OrderTypes] AS (
	SELECT 
		*
	FROM
		[Ingestion].[Orders].[OrderTypes]
)
GO
/****** Object:  View [Resources].[CLN_Dishes]    Script Date: 12.01.2024 16:34:55 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE VIEW [Resources].[CLN_Dishes] AS (
	SELECT 
		*
	FROM
		[Ingestion].Resources.Dishes
)
GO
/****** Object:  View [Resources].[CLN_DishProducts]    Script Date: 12.01.2024 16:34:55 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE VIEW [Resources].[CLN_DishProducts] AS (
	SELECT 
		*
	FROM
		[Ingestion].Resources.DishProducts
)
GO
/****** Object:  View [Resources].[CLN_LocationMenus]    Script Date: 12.01.2024 16:34:55 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE VIEW [Resources].[CLN_LocationMenus] AS (
	SELECT
		*
	FROM
		Ingestion.Restaurants.LocationMenus
)
GO
/****** Object:  View [Resources].[CLN_MenuDishes]    Script Date: 12.01.2024 16:34:56 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE VIEW [Resources].[CLN_MenuDishes] AS (
	SELECT
		*
	FROM
		Ingestion.Resources.MenuDishes
)
GO
/****** Object:  View [Resources].[CLN_Menus]    Script Date: 12.01.2024 16:34:56 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE VIEW [Resources].[CLN_Menus] AS (
	SELECT
		*
	FROM
		Ingestion.Resources.Menus
)
GO
/****** Object:  View [Resources].[CLN_Products]    Script Date: 12.01.2024 16:34:56 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE VIEW [Resources].[CLN_Products] AS (
	SELECT 
		*
	FROM
		[Ingestion].Resources.Products
)
GO
/****** Object:  View [Restaurants].[CLN_Addresses]    Script Date: 12.01.2024 16:34:56 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE VIEW [Restaurants].[CLN_Addresses] AS (
	SELECT 
		*
	FROM
		[Ingestion].Restaurants.Addresses
)
GO
/****** Object:  View [Restaurants].[CLN_Locations]    Script Date: 12.01.2024 16:34:56 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE VIEW [Restaurants].[CLN_Locations] AS (
	SELECT 
		*
	FROM
		[Ingestion].Restaurants.Locations
)
GO
/****** Object:  View [Restaurants].[CLN_Reservations]    Script Date: 12.01.2024 16:34:56 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE VIEW [Restaurants].[CLN_Reservations] AS (
	SELECT
		R.ID as ReservationID
		, R.CustomerID
		, R.Seats AS Seats
		, R.ReservationDate
		, R.ReservationHour
		, CASE WHEN R.DeletedAT IS NOT NULL THEN R.DeletedAt ELSE NULL END CancelledAT
		, R.CreatedAt
	FROM
		Ingestion.Restaurants.Reservations AS R
)
GO
/****** Object:  View [Restaurants].[CLN_TableReservations]    Script Date: 12.01.2024 16:34:56 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE VIEW [Restaurants].[CLN_TableReservations] AS (
	SELECT
		*
	FROM
		Ingestion.Restaurants.TableReservations
)
GO
/****** Object:  View [Restaurants].[CLN_Tables]    Script Date: 12.01.2024 16:34:56 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE VIEW [Restaurants].[CLN_Tables] AS (
	SELECT
		*
	FROM
		Ingestion.Restaurants.Tables
)
GO
/****** Object:  View [Staff].[CLN_Employees]    Script Date: 12.01.2024 16:34:56 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE VIEW [Staff].[CLN_Employees] AS (
	SELECT
		*
	FROM
		Ingestion.Staff.Employees
)
GO
USE [master]
GO
ALTER DATABASE [Cleaned] SET  READ_WRITE 
GO
