USE [master]
GO
/****** Object:  Database [Modeled]    Script Date: 12.01.2024 16:41:13 ******/
CREATE DATABASE [Modeled]
 CONTAINMENT = NONE
 ON  PRIMARY 
( NAME = N'Modeled', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL15.MSSQLSERVER\MSSQL\DATA\Modeled.mdf' , SIZE = 73728KB , MAXSIZE = UNLIMITED, FILEGROWTH = 65536KB )
 LOG ON 
( NAME = N'Modeled_log', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL15.MSSQLSERVER\MSSQL\DATA\Modeled_log.ldf' , SIZE = 73728KB , MAXSIZE = 2048GB , FILEGROWTH = 65536KB )
 WITH CATALOG_COLLATION = DATABASE_DEFAULT
GO
ALTER DATABASE [Modeled] SET COMPATIBILITY_LEVEL = 150
GO
IF (1 = FULLTEXTSERVICEPROPERTY('IsFullTextInstalled'))
begin
EXEC [Modeled].[dbo].[sp_fulltext_database] @action = 'enable'
end
GO
ALTER DATABASE [Modeled] SET ANSI_NULL_DEFAULT OFF 
GO
ALTER DATABASE [Modeled] SET ANSI_NULLS OFF 
GO
ALTER DATABASE [Modeled] SET ANSI_PADDING OFF 
GO
ALTER DATABASE [Modeled] SET ANSI_WARNINGS OFF 
GO
ALTER DATABASE [Modeled] SET ARITHABORT OFF 
GO
ALTER DATABASE [Modeled] SET AUTO_CLOSE OFF 
GO
ALTER DATABASE [Modeled] SET AUTO_SHRINK OFF 
GO
ALTER DATABASE [Modeled] SET AUTO_UPDATE_STATISTICS ON 
GO
ALTER DATABASE [Modeled] SET CURSOR_CLOSE_ON_COMMIT OFF 
GO
ALTER DATABASE [Modeled] SET CURSOR_DEFAULT  GLOBAL 
GO
ALTER DATABASE [Modeled] SET CONCAT_NULL_YIELDS_NULL OFF 
GO
ALTER DATABASE [Modeled] SET NUMERIC_ROUNDABORT OFF 
GO
ALTER DATABASE [Modeled] SET QUOTED_IDENTIFIER OFF 
GO
ALTER DATABASE [Modeled] SET RECURSIVE_TRIGGERS OFF 
GO
ALTER DATABASE [Modeled] SET  ENABLE_BROKER 
GO
ALTER DATABASE [Modeled] SET AUTO_UPDATE_STATISTICS_ASYNC OFF 
GO
ALTER DATABASE [Modeled] SET DATE_CORRELATION_OPTIMIZATION OFF 
GO
ALTER DATABASE [Modeled] SET TRUSTWORTHY OFF 
GO
ALTER DATABASE [Modeled] SET ALLOW_SNAPSHOT_ISOLATION OFF 
GO
ALTER DATABASE [Modeled] SET PARAMETERIZATION SIMPLE 
GO
ALTER DATABASE [Modeled] SET READ_COMMITTED_SNAPSHOT OFF 
GO
ALTER DATABASE [Modeled] SET HONOR_BROKER_PRIORITY OFF 
GO
ALTER DATABASE [Modeled] SET RECOVERY FULL 
GO
ALTER DATABASE [Modeled] SET  MULTI_USER 
GO
ALTER DATABASE [Modeled] SET PAGE_VERIFY CHECKSUM  
GO
ALTER DATABASE [Modeled] SET DB_CHAINING OFF 
GO
ALTER DATABASE [Modeled] SET FILESTREAM( NON_TRANSACTED_ACCESS = OFF ) 
GO
ALTER DATABASE [Modeled] SET TARGET_RECOVERY_TIME = 60 SECONDS 
GO
ALTER DATABASE [Modeled] SET DELAYED_DURABILITY = DISABLED 
GO
ALTER DATABASE [Modeled] SET ACCELERATED_DATABASE_RECOVERY = OFF  
GO
EXEC sys.sp_db_vardecimal_storage_format N'Modeled', N'ON'
GO
ALTER DATABASE [Modeled] SET QUERY_STORE = OFF
GO
USE [Modeled]
GO
/****** Object:  User [NT SERVICE\MSSQLServerOLAPService]    Script Date: 12.01.2024 16:41:13 ******/
CREATE USER [NT SERVICE\MSSQLServerOLAPService] FOR LOGIN [NT SERVICE\MSSQLServerOLAPService] WITH DEFAULT_SCHEMA=[Reports]
GO
ALTER ROLE [db_datareader] ADD MEMBER [NT SERVICE\MSSQLServerOLAPService]
GO
/****** Object:  Schema [Reports]    Script Date: 12.01.2024 16:41:13 ******/
CREATE SCHEMA [Reports]
GO
/****** Object:  Table [Reports].[MDL_CalendarDateDim]    Script Date: 12.01.2024 16:41:13 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Reports].[MDL_CalendarDateDim](
	[calendarDate] [date] NULL,
	[Year] [int] NULL,
	[Month] [int] NULL,
	[Day] [int] NULL,
	[Quarter] [int] NULL,
	[Holiday] [varchar](6) NULL
) ON [PRIMARY]
GO
/****** Object:  Table [Reports].[MDL_CustomerAddressesDim]    Script Date: 12.01.2024 16:41:13 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Reports].[MDL_CustomerAddressesDim](
	[AddressID] [int] NOT NULL,
	[CustomerAddress] [varchar](511) NULL,
	[ZipCode] [char](5) NOT NULL,
	[CityName] [varchar](255) NOT NULL,
	[StateName] [varchar](255) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[AddressID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Reports].[MDL_CustomersDim]    Script Date: 12.01.2024 16:41:13 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Reports].[MDL_CustomersDim](
	[CustomerID] [int] IDENTITY(1,1) NOT NULL,
	[Title] [nvarchar](5) NULL,
	[FirstName] [varchar](50) NOT NULL,
	[Blocked] [bit] NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[CustomerID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Reports].[MDL_DeliveryFacts]    Script Date: 12.01.2024 16:41:13 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Reports].[MDL_DeliveryFacts](
	[DeliveryID] [int] NOT NULL,
	[LocationID] [smallint] NOT NULL,
	[DeliveryEmployeeID] [int] NOT NULL,
	[CustomerID] [int] NULL,
	[DeliveryStatus] [varchar](25) NOT NULL,
	[DeliveryAddressId] [int] NULL,
	[Price] [money] NULL,
	[Quantity] [int] NULL,
	[DelliveredTime] [datetime] NULL,
PRIMARY KEY CLUSTERED 
(
	[DeliveryID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Reports].[MDL_DeliveryItemFacts]    Script Date: 12.01.2024 16:41:13 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Reports].[MDL_DeliveryItemFacts](
	[DeliveryItemId] [int] NOT NULL,
	[DishID] [int] NOT NULL,
	[DeliveryDriverID] [int] NOT NULL,
	[DeliveryQuantity] [smallint] NOT NULL,
	[ExepctedQuantity] [smallint] NULL,
	[CancelledAt] [datetime2](7) NULL,
	[FullPrice] [money] NULL
) ON [PRIMARY]
GO
/****** Object:  Table [Reports].[MDL_DishProductDim]    Script Date: 12.01.2024 16:41:13 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Reports].[MDL_DishProductDim](
	[ID] [int] NOT NULL,
	[DishID] [int] NOT NULL,
	[Name] [varchar](255) NOT NULL,
	[ProductPrice] [int] NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Reports].[MDL_LocationMenuFacts]    Script Date: 12.01.2024 16:41:13 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Reports].[MDL_LocationMenuFacts](
	[ID] [int] NOT NULL,
	[LocationID] [smallint] NOT NULL,
	[MenuID] [int] NOT NULL,
	[MenuValidFrom] [date] NOT NULL,
	[MenuValidTo] [date] NULL,
	[IntroductionDelay] [int] NULL,
	[ExecutiveChefID] [int] NOT NULL,
	[DishesCount] [int] NULL,
	[AppetizersCount] [int] NULL,
	[DessertsCount] [int] NULL,
	[DrinksCount] [int] NULL,
	[MainCoursesCount] [int] NULL
) ON [PRIMARY]
GO
/****** Object:  Table [Reports].[MDL_LocationsDim]    Script Date: 12.01.2024 16:41:13 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Reports].[MDL_LocationsDim](
	[LocationID] [smallint] NOT NULL,
	[LocationAddress] [varchar](511) NULL,
	[ZipCode] [char](5) NOT NULL,
	[CityName] [varchar](255) NOT NULL,
	[StateName] [varchar](255) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[LocationID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Reports].[MDL_OrderItemFacts]    Script Date: 12.01.2024 16:41:13 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Reports].[MDL_OrderItemFacts](
	[OrderItemId] [int] NOT NULL,
	[DishID] [int] NOT NULL,
	[LocationID] [smallint] NOT NULL,
	[CustomerID] [int] NULL,
	[Quantity] [smallint] NOT NULL,
	[UnitPrice] [smallmoney] NOT NULL,
	[OrderType] [varchar](25) NOT NULL,
	[OrderDate] [date] NULL,
	[OrderTime] [time](0) NULL,
PRIMARY KEY CLUSTERED 
(
	[OrderItemId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Reports].[MDL_OrdersFacts]    Script Date: 12.01.2024 16:41:13 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Reports].[MDL_OrdersFacts](
	[OrderID] [int] NOT NULL,
	[LocationID] [smallint] NOT NULL,
	[CustomerID] [int] NULL,
	[IsReservation] [int] NOT NULL,
	[OrderStatus] [varchar](25) NOT NULL,
	[OrderType] [varchar](25) NOT NULL,
	[DishesBought] [int] NULL,
	[FullPrice] [money] NULL,
	[OrderDate] [date] NULL,
	[OrderTime] [time](0) NULL,
PRIMARY KEY CLUSTERED 
(
	[OrderID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [Reports].[MDL_ReservationFacts]    Script Date: 12.01.2024 16:41:13 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Reports].[MDL_ReservationFacts](
	[ReservationID] [int] NOT NULL,
	[CustomerID] [int] NOT NULL,
	[LocationID] [smallint] NOT NULL,
	[Seats] [smallint] NOT NULL,
	[ReservationDate] [date] NOT NULL,
	[ReservationHour] [time](7) NOT NULL,
	[CancelledAt] [datetime2](7) NULL,
	[CreatedAt] [datetime2](7) NOT NULL
) ON [PRIMARY]
GO
/****** Object:  Table [Reports].[MDL_WeatherDim]    Script Date: 12.01.2024 16:41:13 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Reports].[MDL_WeatherDim](
	[date] [date] NULL,
	[CityID] [int] IDENTITY(1,1) NOT NULL,
	[Weather] [varchar](10) NOT NULL
) ON [PRIMARY]
GO
/****** Object:  Table [Reports].[MLD_EmployeesDim]    Script Date: 12.01.2024 16:41:13 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Reports].[MLD_EmployeesDim](
	[EmployeeID] [int] IDENTITY(1,1) NOT NULL,
	[FirstName] [varchar](255) NOT NULL,
	[LastName] [varchar](255) NOT NULL,
	[BirthDate] [date] NOT NULL
) ON [PRIMARY]
GO
USE [master]
GO
ALTER DATABASE [Modeled] SET  READ_WRITE 
GO
